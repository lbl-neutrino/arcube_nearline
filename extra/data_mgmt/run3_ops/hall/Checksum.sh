#!/usr/bin/env bash
#
# Checksum.sh
#
# Compare Adler-32 checksums from a checksum list against MetaCat and Rucio.
#
# Usage:
#   ./Checksum.sh checksums.txt
#
# Optional namespace/scope override:
#   METACAT_NAMESPACE=my-namespace ./Checksum.sh checksums.txt
#
# The input file must contain lines like:
#   35ce3166 /some/remote/path/binary-0060006-...hdf5.copied
#

set -u
set -o pipefail

INPUT_FILE="${1:-}"

# This is both the MetaCat namespace and the Rucio scope in the example
# workflow. Override it through METACAT_NAMESPACE if needed.
NAMESPACE="${METACAT_NAMESPACE:-neardet-2x2-lar-charge}"

if [[ -z "$INPUT_FILE" ]]; then
    echo "Usage: $0 <checksums.txt>" >&2
    exit 2
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: input file does not exist: $INPUT_FILE" >&2
    exit 2
fi

for command in metacat rucio; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Error: '$command' is not available in PATH." >&2
        exit 2
    fi
done

total=0
passed=0
failed=0

echo "Input file : $INPUT_FILE"
echo "Namespace  : $NAMESPACE"
echo

while IFS=$' \t' read -r local_checksum remote_path; do

    # Ignore blank lines.
    [[ -z "${local_checksum:-}" ]] && continue

    total=$((total + 1))

    local_checksum="$(tr '[:upper:]' '[:lower:]' <<< "$local_checksum")"

    if [[ ! "$local_checksum" =~ ^[0-9a-f]{8}$ ]]; then
        echo "[FAIL] Invalid local Adler-32 checksum on input line $total:" >&2
        echo "       $local_checksum $remote_path" >&2
        failed=$((failed + 1))
        continue
    fi

    # Keep only the filename; no remote directory is included in the DID.
    filename="$(basename "$remote_path")"

    if [[ "$filename" != *.copied ]]; then
        echo "[FAIL] File does not end in .copied:" >&2
        echo "       $filename" >&2
        failed=$((failed + 1))
        continue
    fi

    # Remove only the final ".copied" suffix.
    catalog_filename="${filename%.copied}"
    did="${NAMESPACE}:${catalog_filename}"

    printf '[%d] %s\n' "$total" "$did"

    #
    # MetaCat query
    #
    metacat_output=""
    if ! metacat_output="$(metacat file show "$did" 2>&1)"; then
        echo "  [FAIL] MetaCat query failed." >&2
        echo "         $metacat_output" >&2
        failed=$((failed + 1))
        continue
    fi

    metacat_checksum="$(
        sed -nE \
            's/^[[:space:]]*adler32[[:space:]]*:[[:space:]]*([[:xdigit:]]{8})[[:space:]]*$/\1/p' \
            <<< "$metacat_output" \
        | head -n 1 \
        | tr '[:upper:]' '[:lower:]'
    )"

    if [[ -z "$metacat_checksum" ]]; then
        echo "  [FAIL] No Adler-32 checksum found in MetaCat output." >&2
        failed=$((failed + 1))
        continue
    fi

    #
    # Rucio query
    #
    rucio_output=""
    if ! rucio_output="$(rucio replica list file "$did" 2>&1)"; then
        echo "  [FAIL] Rucio query failed." >&2
        echo "         $rucio_output" >&2
        failed=$((failed + 1))
        continue
    fi

    # Extract the ADLER32 column from table rows such as:
    # | scope | name | filesize | 35ce3166 | RSE: replica |
    #
    # sort -u ensures that all reported replicas must agree.
    rucio_checksums="$(
        grep -Eo '\|[[:space:]]*[[:xdigit:]]{8}[[:space:]]*\|' <<< "$rucio_output" \
        | sed -E 's/[|[:space:]]//g' \
        | tr '[:upper:]' '[:lower:]' \
        | sort -u
    )"

    if [[ -z "$rucio_checksums" ]]; then
        echo "  [FAIL] No Adler-32 checksum found in Rucio output." >&2
        failed=$((failed + 1))
        continue
    fi

    #
    # Comparison
    #
    # rucio_checksums must contain exactly one unique checksum, and it must
    # match both the local and MetaCat values.
    if [[ "$local_checksum" == "$metacat_checksum" &&
          "$rucio_checksums" == "$local_checksum" ]]; then

        echo "  [PASS] local=$local_checksum  metacat=$metacat_checksum  rucio=$rucio_checksums"
        passed=$((passed + 1))
    else
        echo "  [FAIL] Checksum mismatch:" >&2
        echo "         local   : $local_checksum" >&2
        echo "         metacat : $metacat_checksum" >&2
        echo "         rucio   : ${rucio_checksums//$'\n'/, }" >&2
        failed=$((failed + 1))
    fi

done < "$INPUT_FILE"

echo
echo "Summary: total=$total  passed=$passed  failed=$failed"

if (( failed > 0 )); then
    exit 1
fi

exit 0
