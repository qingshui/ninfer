#pragma once

#include "runtime/contract/resources.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ninfer::runtime {

// Compressed edges refer to a surviving native checkpoint's semantic digest path. The model
// verifies exact token/position/media identity before a returned checkpoint can be used.
template <class Model>
class PrefixIndex {
public:
    using Handle  = typename Model::CheckpointHandle;
    using Session = typename Model::Session;
    using Base    = typename Model::RequestBasePlan;

    void insert(Session& session, Handle handle, std::uint32_t frontier, CheckpointRole role) {
        if (insert_at(session, root_, handle, frontier, role)) {
            ++counters(role).retained;
            ++counters(role).published;
        }
    }

    // Walks only the path of the node that holds `handle`, from that node up to the root.
    void erase(Handle handle, bool reclaimed = false) {
        if (handle.index >= locations_.size()) return;
        auto& slot = locations_[handle.index];
        for (auto found = find_location(slot, handle); found != slot.end();
             found      = find_location(slot, handle)) {
            Node* const holder = found->node;
            slot.erase(found);
            std::erase_if(resolve(holder).checkpoints, [&](const auto& entry) {
                if (entry.handle != handle) return false;
                auto& value = counters(entry.role);
                --value.retained;
                ++value.released;
                if (reclaimed) ++value.reclaimed;
                return true;
            });
            restore_path(holder);
        }
    }

    [[nodiscard]] const PrefixCacheStatistics& statistics() const noexcept { return statistics_; }

    void record_adoption(CheckpointRole role, std::uint64_t tokens) noexcept {
        auto& value = counters(role);
        ++value.adopted;
        value.reused_tokens += tokens;
    }

    [[nodiscard]] std::vector<Handle> matches(const Session& session, const Base& base) const {
        std::vector<Handle> result;
        const Node* current = &root_;
        while (current) {
            for (const auto& entry : current->checkpoints) result.push_back(entry.handle);
            const Node* next = nullptr;
            for (const auto& child : current->children) {
                if (child->frontier > base.summary().prompt_tokens) { continue; }
                const auto key = base.prefix_shortlist_key(child->frontier);
                if (key && *key == session.checkpoint_key(child->representative, child->frontier)) {
                    next = child.get();
                    break;
                }
            }
            current = next;
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

    void clear() noexcept {
        for (auto* value : {&statistics_.input_replay, &statistics_.long_anchor,
                            &statistics_.shared_prefix, &statistics_.continuation}) {
            value->released += value->retained;
            value->retained = 0;
        }
        root_ = Node{};
        locations_.clear();
    }

private:
    struct Entry {
        Handle handle;
        CheckpointRole role;
    };

    struct Node {
        std::uint32_t frontier = 0;
        Handle representative{};
        std::vector<Entry> checkpoints;
        std::vector<std::unique_ptr<Node>> children;
        // The node whose children hold this one; null for a child of the root.
        Node* parent = nullptr;
    };

    // The node that holds a handle's entry, null for the root. Kept per checkpoint slot: a slot
    // can hold an old generation until the manager forgets it.
    struct Location {
        Handle handle;
        Node* node;
    };

    PrefixCacheCounters& counters(CheckpointRole role) noexcept {
        switch (role) {
        case CheckpointRole::InputReplay:
            return statistics_.input_replay;
        case CheckpointRole::LongAnchor:
            return statistics_.long_anchor;
        case CheckpointRole::SharedPrefix:
            return statistics_.shared_prefix;
        case CheckpointRole::Continuation:
            return statistics_.continuation;
        }
        std::terminate();
    }

    Node& resolve(Node* node) noexcept { return node ? *node : root_; }
    Node* address(Node& node) noexcept { return &node == &root_ ? nullptr : &node; }

    static auto find_location(std::vector<Location>& slot, Handle handle) {
        return std::find_if(slot.begin(), slot.end(),
                            [&](const auto& location) { return location.handle == handle; });
    }

    void locate(Handle handle, Node* node) {
        if (handle.index >= locations_.size()) locations_.resize(handle.index + std::size_t{1});
        locations_[handle.index].push_back({handle, node});
    }

    void relocate(Handle handle, const Node* from, Node* to) {
        for (auto& location : locations_[handle.index])
            if (location.handle == handle && location.node == from) location.node = to;
    }

    bool insert_at(Session& session, Node& node, Handle handle, std::uint32_t frontier,
                   CheckpointRole role) {
        if (node.frontier == frontier) {
            const bool added =
                std::none_of(node.checkpoints.begin(), node.checkpoints.end(),
                             [&](const auto& entry) { return entry.handle == handle; });
            if (added) {
                node.checkpoints.push_back({handle, role});
                locate(handle, address(node));
            }
            node.representative = handle;
            return added;
        }
        for (auto& child : node.children) {
            if (session.checkpoint_key(child->representative, node.frontier + 1U) !=
                session.checkpoint_key(handle, node.frontier + 1U)) {
                continue;
            }
            std::uint32_t low = node.frontier + 1U, high = std::min(frontier, child->frontier);
            while (low < high) {
                const auto middle = low + (high - low + 1U) / 2U;
                if (session.checkpoint_key(child->representative, middle) ==
                    session.checkpoint_key(handle, middle)) {
                    low = middle;
                } else {
                    high = middle - 1U;
                }
            }
            if (low < child->frontier) {
                auto branch            = std::make_unique<Node>();
                branch->frontier       = low;
                branch->representative = child->representative;
                branch->parent         = address(node);
                child->parent          = branch.get();
                branch->children.push_back(std::move(child));
                child = std::move(branch);
            }
            return insert_at(session, *child, handle, frontier, role);
        }
        auto child            = std::make_unique<Node>();
        child->frontier       = frontier;
        child->representative = handle;
        child->parent         = address(node);
        child->checkpoints.push_back({handle, role});
        locate(handle, child.get());
        node.children.push_back(std::move(child));
        return true;
    }

    // After an entry left `node`, the node and its ancestors (the root excepted) regain the shape
    // insertion builds: an empty node goes, a node left with one child and no entry of its own
    // takes that child's place, and each keeps a representative from its own subtree.
    void restore_path(Node* node) {
        while (node) {
            Node* const parent = node->parent;
            auto& siblings     = resolve(parent).children;
            if (node->checkpoints.empty() && node->children.empty()) {
                const auto found =
                    std::find_if(siblings.begin(), siblings.end(),
                                 [&](const auto& child) { return child.get() == node; });
                if (found == siblings.end())
                    throw std::logic_error("prefix index node is absent from its parent");
                siblings.erase(found);
            } else {
                if (node->checkpoints.empty() && node->children.size() == 1) {
                    const auto child = std::move(node->children.front());
                    *node            = std::move(*child);
                    node->parent     = parent;
                    for (const auto& entry : node->checkpoints)
                        relocate(entry.handle, child.get(), node);
                    for (const auto& grandchild : node->children) grandchild->parent = node;
                }
                node->representative = !node->checkpoints.empty()
                                           ? node->checkpoints.front().handle
                                           : node->children.front()->representative;
            }
            node = parent;
        }
    }

    Node root_;
    std::vector<std::vector<Location>> locations_;
    PrefixCacheStatistics statistics_;
};
} // namespace ninfer::runtime
