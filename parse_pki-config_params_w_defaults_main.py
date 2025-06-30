import os
import re
import json
from collections import defaultdict

dogtag_repo = "/home/taherrin/taherrin_dev/dogtagpki/pki/"
cs_cfg_file = "/home/taherrin/taherrin_dev/dogtagpki/pki/base/ca/shared/conf/CS.cfg"
file_extentions = ['.java', '.md', '.sh', '.py', '.yaml', '.yml', '.cfg', '.adoc',
                     '.txt', '.conf','.properties', '.ldif','.html', '.profile', '.json']
context_lines = 4  # Lines before/after usage
parsed_data_output = "config_context_dataset_test.jsonl"

# Subset of known keys. If empty all keys found in defaults can be used instead
known_config_params = [
    "debug.level",
    "jobsScheduler.enabled",
    "internaldb.ldapconn.secureConn",
    "authType",
    "machineName",
    "ca.cert.list",
    "ca.scep.enable",
    "ca.crl.MasterCRL.enable",
    "ca.crl.MasterCRL.signingAlgorithm",
    "ca.crl.MasterCRL.alwaysUpdate",
    "ca.crl.MasterCRL.autoUpdateInterval",
    "dbs.enableSerialManagement",
    "dbs.ldap",
    "internaldb.ldapauth.authtype",
    "internaldb.ldapconn.port",
    "jobsScheduler.interval",
    "log.instance.SignedAudit.logSigning",
    "selftests.container.instance.SystemCertsVerification",
    "internaldb.ldapauth.clientCertNickname",
    "ca.publish.queue.enable",
    "keys.rsa.keysize.default",
    "ca.profiles.defaultSigningAlgsAllowed"
]

# Regex to detect key=value lines
key_value_regex = re.compile(r'^[#;]?\s*([a-zA-Z0-9._-]+)\s*=\s*(.*)$')

def extract_defaults(config_file_path):
    defaults = {}

    try:
        with open(config_file_path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = key_value_regex.match(line.strip())
                if match:
                    key, val = match.groups()
                    defaults[key] = val
    except Exception as e:
        print(f"Error reading {config_file_path}: {e}")

    return defaults

def collect_references(config_keys, root_dir, exts, context_lines=4):
    grouped = defaultdict(list)
    key_patterns = {k: re.compile(re.escape(k)) for k in config_keys}
    omit_directory = {'pytest-ansible'}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in omit_directory]

        for filename in filenames:
            if any(filename.endswith(ext) for ext in exts):
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                except Exception as e:
                    print(f"Skipped {filepath}: {e}")
                    continue

                for i, line in enumerate(lines):
                    for key, pattern in key_patterns.items():
                        if pattern.search(line):
                            start = max(0, i - context_lines)
                            end = min(len(lines), i + context_lines + 1)
                            snippet = ''.join(lines[start:end]).strip()
                            grouped[key].append({
                                "file": filepath,
                                "line": i + 1,
                                "snippet": snippet
                            })

    return grouped

def create_dataset(grouped_contexts, defaults_dict, output_path):
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for key, contexts in grouped_contexts.items():
            if not contexts:
                continue

            # Compose multi-reference context block
            context_block = ""
            for ref in contexts:
                context_block += f"- File: {ref['file']}, Line {ref['line']}:\n{ref['snippet']}\n\n"

            default_val = defaults_dict.get(key, "Unknown")

            prompt = (
                f"Config key: {key}\n\n"
                f"Default value: {default_val}\n\n"
                f"Context:\n{context_block.strip()}\n\n"
                f"What does this config key do?"
            )

            manual_def = "Fill in definition here"

            jsonl_obj = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": manual_def}
                ]
            }
            out_file.write(json.dumps(jsonl_obj) + '\n')

    print(f"JSONL dataset saved to {output_path}")


# Step 1: Extract defaults from config files
print("Extracting default values from config files...")
defaults = extract_defaults(cs_cfg_file)
print(f"Found {len(defaults)} default key=value pairs.")

# Decide which keys to collect usage for:
# Use known_config_params if defined, else all keys found in defaults
if known_config_params:
    keys_to_search = known_config_params
else:
    keys_to_search = list(defaults.keys())

print(f"Searching for usages of {len(keys_to_search)} config keys...")

# Step 2: Collect usage contexts
grouped = collect_references(keys_to_search, dogtag_repo, file_extentions, context_lines)

# Step 3: Build JSONL for fine-tuning
create_dataset(grouped, defaults, parsed_data_output)
