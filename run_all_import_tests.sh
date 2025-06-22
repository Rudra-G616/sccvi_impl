#!/bin/bash

# run_all_import_tests.sh
# Script to run all import tests and generate a combined report

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}      sccvi_impl Import Tests - Combined Report      ${NC}"
echo -e "${BLUE}====================================================${NC}"
echo ""

# Create a results directory
RESULTS_DIR="import_test_results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

# Function to run a test and capture its output
run_test() {
    local test_name="$1"
    local test_script="$2"
    local output_file="$RESULTS_DIR/${test_name}_${TIMESTAMP}.log"
    
    echo -e "${YELLOW}Running $test_name...${NC}"
    echo "$ python $test_script"
    
    # Run the test and capture output, exit code
    python "$test_script" > "$output_file" 2>&1
    local exit_code=$?
    
    # Print status based on exit code
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ $test_name passed!${NC}"
    else
        echo -e "${RED}✗ $test_name failed! (Exit code: $exit_code)${NC}"
        echo -e "${YELLOW}See full output in:${NC} $output_file"
        
        # Show a brief summary of errors
        echo ""
        echo -e "${YELLOW}Error summary from $test_name:${NC}"
        grep -A 2 "Failed to import" "$output_file" | head -n 15
        if [ $(grep -c "Failed to import" "$output_file") -gt 5 ]; then
            echo -e "${YELLOW}... and more errors (see log file for complete list)${NC}"
        fi
        echo ""
    fi
    
    return $exit_code
}

# Run all three tests
test1_result=0
test2_result=0
test3_result=0

run_test "Basic Import Test" "test_imports.py"
test1_result=$?

run_test "Pytest Import Test" "test_imports_pytest.py"
test2_result=$?

run_test "Directory-based Import Test" "test_all_imports.py"
test3_result=$?

# Print summary
echo ""
echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}                  Summary                           ${NC}"
echo -e "${BLUE}====================================================${NC}"

if [ $test1_result -eq 0 ] && [ $test2_result -eq 0 ] && [ $test3_result -eq 0 ]; then
    echo -e "${GREEN}All import tests passed successfully!${NC}"
    final_result=0
else
    echo -e "${RED}Some import tests failed:${NC}"
    [ $test1_result -ne 0 ] && echo -e "${RED}✗ Basic Import Test failed${NC}"
    [ $test2_result -ne 0 ] && echo -e "${RED}✗ Pytest Import Test failed${NC}"
    [ $test3_result -ne 0 ] && echo -e "${RED}✗ Directory-based Import Test failed${NC}"
    
    echo ""
    echo -e "${YELLOW}See detailed test logs in the $RESULTS_DIR directory${NC}"
    final_result=1
fi

echo ""
echo -e "${BLUE}Test logs saved to: $RESULTS_DIR${NC}"
exit $final_result
