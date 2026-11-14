#include "options.h"

#include <charconv>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace ninfer::perplexity {
namespace {

constexpr product::CommandLineOption kOptions[] = {
    {"--corpus"}, {"--text"}, {"--output"}, {"--context"}, {"--stride"}, {"--quick", false},
};

std::uint32_t count(const product::CommandLineArgument& argument) {
    std::uint32_t value     = 0;
    const auto& text        = argument.value;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::invalid_argument(argument.flag + " requires an unsigned integer");
    return value;
}

std::filesystem::path path(const product::CommandLineArgument& argument) {
    if (argument.value.empty()) throw std::invalid_argument(argument.flag + " must not be empty");
    return std::filesystem::absolute(argument.value).lexically_normal();
}

// The fixed quick corpus shipped next to the executable's build tree:
// <build>/apps/ninfer-perplexity -> <repo>/eval/corpora/perplexity-1m/manifest.json.
std::optional<std::filesystem::path> bundled_quick_corpus() {
    std::error_code error;
    const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
    if (error) return std::nullopt;
    auto manifest = std::filesystem::weakly_canonical(
        executable.parent_path() / "../../../eval/corpora/perplexity-1m/manifest.json", error);
    if (error || !std::filesystem::is_regular_file(manifest)) return std::nullopt;
    return manifest;
}

} // namespace

Options parse_options(int argc, char** argv) {
    auto parsed =
        product::parse_launch_options(argc, argv, product::EntryPoint::Perplexity, kOptions);
    Options options;
    options.help_requested = parsed.help_requested;
    options.launch         = std::move(parsed.launch);
    if (options.help_requested) return options;
    for (const auto& argument : parsed.extra_arguments) {
        if (argument.flag == "--corpus")
            options.corpus = path(argument);
        else if (argument.flag == "--text")
            options.text = path(argument);
        else if (argument.flag == "--output")
            options.output = path(argument);
        else if (argument.flag == "--context")
            options.context = count(argument);
        else if (argument.flag == "--stride")
            options.stride = count(argument);
        else if (argument.flag == "--quick")
            options.quick = true;
    }
    if (!options.corpus && !options.text && options.quick)
        options.corpus = bundled_quick_corpus();
    if (!options.corpus && !options.text && options.quick)
        throw std::invalid_argument(
            "--quick without --corpus requires the bundled perplexity-1m corpus next to the "
            "executable; pass --corpus explicitly");
    if (options.corpus.has_value() == options.text.has_value())
        throw std::invalid_argument("exactly one of --corpus and --text is required");
    if (options.quick && !options.corpus) throw std::invalid_argument("--quick requires --corpus");
    if (options.context < 2 || !options.stride || options.stride >= options.context)
        throw std::invalid_argument("context/stride must satisfy context>=2 and 1<=stride<context");
    if (!options.launch.provenance.fields.contains("engine.max_context"))
        options.launch.engine.max_context = options.context;
    if (options.context > options.launch.engine.max_context)
        throw std::invalid_argument("--context scoring window exceeds engine.max_context; "
                                    "increase --max-context or reduce --context");
    return options;
}

std::string usage_text() {
    return product::startup_usage_text("ninfer-perplexity", product::EntryPoint::Perplexity) + R"(
perplexity:
  --corpus FILE / --text FILE
      Select a corpus manifest or one UTF-8 file. Exactly one is required.
  --quick
      Use the fixed corpus subset; requires --corpus. Without --corpus or --text,
      falls back to the bundled eval/corpora/perplexity-1m corpus next to the executable.
  --context N        Scoring window length (default 4096).
  --stride N         Newly scored targets per later window (default 2048).
  --output DIRECTORY Write report.json here; the directory must be absent or empty.
                     Default: profiles/perplexity/<model>/<format>/<kv>/<corpus>/<mode>/<time>.

The scoring window and stride define the metric. They are independent of prefill chunk size.
Engine max-context defaults to the window unless set in TOML or with --max-context;
the scoring window must fit that limit. KV defaults to fp8.
Public startup/deployment settings come from TOML; corpus and window options are command-line only.
Generation-only TOML settings do not apply. Engine causal scoring uses fresh Text State/KV for
each window, the main head, one request, and no speculative decoding, Graph or prefix reuse.
Every token after the first in each stream is scored exactly once. Reports include actual rank
resources and token-weighted NLL/PPL; summaries use stdout and progress uses stderr.
)";
}

} // namespace ninfer::perplexity
