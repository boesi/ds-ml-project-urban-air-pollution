# Install nbcommands if not already installed. More info: https://github.com/vinayak-mehta/nbcommands
if ! command -v nbgrep &> /dev/null; then
    echo "nbcommands is not installed. Installing..."
    pip install nbcommands
fi

# Specify the path for the output Python file
mkdir -p ./.github/workflows/testing
touch ./.github/workflows/testing/test_import_libraries.py
PYTHON_FILE="./.github/workflows/testing/test_import_libraries.py"

# Remove the output file if it already exists
rm -f "$PYTHON_FILE"

# Search for *.ipynb files recursively in all folders
find . -type f -name '*.ipynb' -print0 |
    while IFS= read -r -d '' file; do
        # Extract import statements from Jupyter notebooks               
        # new one catch the import xx.x as y and removing the space
        # nbgrep "(?:from\s+(\w+(?:\.\w+)*)\s+import\s+(\*|\w+)(?:\s+as\s+\w+)?(?:,s*\w+(?:\\s+as\s+\w+)?)*|import\s+(\w+(?:\.\w+)*)\s+as\s+\w+(?:,\s*\w+(?:\s+as\s+\w+)?)*)" "$file" | grep -v 'nbgrep:' | awk -F ':' '{sub(/^[^:]*:[^:]*:line [0-9]+:/, " ", $0)}1'| sed 's/^[[:space:]]*//'
        nbgrep "(?:from\s+(\w+(?:\.\w+)*)\s+import\s+(\*|\w+)(?:\s+as\s+\w+)?(?:,\s*\w+(?:\s+as\s+\w+)?)*|import\s+(\w+(?:\.\w+)*)\s+as\s+\w+(?:,\s*\w+(?:\s+as\s+\w+)?)*)|(?:^|\n)import\s+(\w+(?:\.\w+)*)\s+" "$file" | grep -v 'nbgrep:' | awk -F ':' '{sub(/^[^:]*:[^:]*:line [0-9]+:/, " ", $0)}1'| sed 's/^[[:space:]]*//'


    done | sort -u > "$PYTHON_FILE"
echo "<<<<<<< All extract import statements from Jupyter notebooks >>>>>>"
cat  "$PYTHON_FILE"


# Generate test function in test.py
# echo "#!/usr/bin/env python" > "$PYTHON_FILE.tmp"
echo "" >> "$PYTHON_FILE.tmp"
echo "def test_import_libraries():" >> "$PYTHON_FILE.tmp"
echo "    try:" >> "$PYTHON_FILE.tmp"
grep  "import" "$PYTHON_FILE" | sed 's/^/        /' >> "$PYTHON_FILE.tmp"
echo "    except ImportError as e:" >> "$PYTHON_FILE.tmp"
echo "        # If any of the libraries cannot be imported, the test will fail" >> "$PYTHON_FILE.tmp"
echo "        assert False, f\"Failed to import library: {e}\"" >> "$PYTHON_FILE.tmp"
echo "" >> "$PYTHON_FILE.tmp"
echo "    # If all libraries are imported successfully, the test passes" >> "$PYTHON_FILE.tmp"
echo "    assert True" >> "$PYTHON_FILE.tmp"

# Move the temporary file to the final location
mv "$PYTHON_FILE.tmp" "$PYTHON_FILE"
cat  "$PYTHON_FILE"
