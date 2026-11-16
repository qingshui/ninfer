#pragma once

#include "ninfer/types.h"
#include "product/logging/logging.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::product {

enum class EntryPoint { Cli, Serve, Bench, Perplexity };
enum class ConfigSourceKind { Default, File, CommandLine };

struct ConfigSource {
    ConfigSourceKind kind = ConfigSourceKind::Default;
    std::uint32_t line    = 0;
    std::uint32_t column  = 0;
    int argument          = 0;
};

struct StartupProvenance {
    std::filesystem::path config_path;
    std::vector<std::string> argv;
    std::map<std::string, ConfigSource, std::less<>> fields;
};

struct GenerationDefaults {
    std::uint32_t max_tokens = 128;
    std::optional<bool> enable_thinking;
    std::optional<bool> preserve_thinking;
    std::optional<std::uint32_t> thinking_budget;
    // A missing seed requests a fresh seed when each request is formed.
    SamplingOverrides sampling;
};

struct StartupLoggingOptions {
    LogLevel level                  = LogLevel::Info;
    std::uint32_t stats_interval_ms = 5000;
    std::filesystem::path request_jsonl;
};

struct LaunchOptions {
    EngineOptions engine;
    GenerationDefaults generation;
    StartupLoggingOptions logging;
    StartupProvenance provenance;
};

[[nodiscard]] LaunchOptions default_launch_options(EntryPoint entry);
[[nodiscard]] std::uint64_t random_sampling_seed();
[[nodiscard]] DeploymentOptions read_deployment_file(const std::filesystem::path&);

enum class ConstraintKind { None, Grammar, JsonObject, JsonSchema, Regex, Choice };

struct ConstraintOptions {
    ConstraintKind kind = ConstraintKind::None;
    std::filesystem::path path;
    std::string regex;
    std::vector<std::string> choices;
};

struct CliOptions {
    bool help_requested  = false;
    LaunchOptions launch = default_launch_options(EntryPoint::Cli);
    std::string prompt;
    std::filesystem::path messages_path;
    std::optional<ReasoningEffort> reasoning_effort;
    bool raw_output      = false;
    bool print_token_ids = false;
    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;
    ConstraintOptions constraint;
};

inline constexpr std::uint32_t kDefaultServeMaxTokens     = 8192;
inline constexpr std::size_t kDefaultMaxRequestBytes      = 384ULL << 20;
inline constexpr std::size_t kDefaultResponseStoreRecords = 1024;
inline constexpr std::size_t kDefaultResponseStoreBytes   = 256ULL << 20;
// Serving admission defaults: fill every decode lane the engine supports and hold
// arriving request bursts in the pending queue (a client disconnect still cancels) rather
// than answering 429 while lanes free up moments later. Pending slots count sequences, so
// the pool covers a full n>1 concurrency wave (for example 130 clients x n=4 = 520).
// The serving default selects automatic lane fitting: zero delegates the lane count to the
// Engine, which resolves it from its context token capacity and reports the fitted value
// through its options and description. Explicit TOML/CLI values still override the default.
inline constexpr std::uint32_t kDefaultServeMaxConcurrency      = 0;
inline constexpr std::uint32_t kDefaultServeMaxPendingRequests  = 1024;
// Admission waits queue rather than reject; the deadline only bounds how long a request may sit
// waiting for admission. It sits just under common client request deadlines (the OpenAI SDK
// default is ten minutes) so the server does not reject a request the client would still have
// waited for; at a healthy drain rate the queue clears in seconds and the deadline never fires.
inline constexpr std::uint32_t kDefaultServePendingTimeoutMs    = 590000;

struct ServeOptions {
    bool help_requested  = false;
    LaunchOptions launch = default_launch_options(EntryPoint::Serve);
    std::string host     = "127.0.0.1";
    int port             = 8080;
    std::string api_key;
    std::optional<std::string> model_id_override;
    bool enable_cors                       = false;
    std::size_t max_request_bytes          = kDefaultMaxRequestBytes;
    std::size_t response_store_max_records = kDefaultResponseStoreRecords;
    std::size_t response_store_max_bytes   = kDefaultResponseStoreBytes;
};

[[nodiscard]] CliOptions parse_cli_options(int argc, char** argv);
[[nodiscard]] ServeOptions parse_serve_options(int argc, char** argv);

struct CommandLineOption {
    std::string_view flag;
    bool takes_value = true;
};

struct CommandLineArgument {
    std::string flag;
    std::string value;
};

struct ParsedLaunchOptions {
    bool help_requested = false;
    LaunchOptions launch;
    std::vector<CommandLineArgument> extra_arguments;
};

// Parse shared startup settings and preserve application-owned arguments in input order.
[[nodiscard]] ParsedLaunchOptions
parse_launch_options(int argc, char** argv, EntryPoint entry,
                     std::span<const CommandLineOption> extra_options = {});
[[nodiscard]] std::string startup_usage_text(std::string_view executable, EntryPoint entry);

using ConfigurationValue = std::variant<bool, std::uint64_t, double, std::string,
                                        std::vector<std::uint32_t>, std::vector<std::string>>;
enum class ConfigurationUnit { None, Bytes, Milliseconds };

struct ConfigurationItem {
    std::string key;
    ConfigurationValue value;
    ConfigSource source;
    ConfigurationUnit unit = ConfigurationUnit::None;
};

struct StartupConfigurationView {
    EntryPoint entry;
    std::filesystem::path config_path;
    std::vector<std::string> argv;
    std::vector<ConfigurationItem> items;
    std::map<std::string, std::string> environment;
};

[[nodiscard]] StartupConfigurationView configuration_view(const CliOptions& options);
[[nodiscard]] StartupConfigurationView configuration_view(const ServeOptions& options);
[[nodiscard]] StartupConfigurationView configuration_view(const LaunchOptions& options,
                                                          EntryPoint entry);

} // namespace ninfer::product
