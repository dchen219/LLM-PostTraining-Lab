"""The frozen verifier. Changing the verifier means a new VERSION, new hashes and a documented reason; never edit it between runs that are compared."""
VERSION = "2.0.0"
# Release identity. Every run in results/run_records was produced with ORIGINAL_SHA256 (original package layout). The two differ only
# because posttrain/verify.py now imports the unchanged grader from posttrain/verifiers/; scripts/verify_code_lineage.py
# reconstructs the original files from this release and checks both hashes.
ORIGINAL_SHA256 = dict(verify_py="b905bcaa4945be2e94286664a5fe9659c30c364ca973b4d90007463de89df106", grader_py="f4fa06fed4950d6bd66a2fc40faaa67943327c2064e9d7ced1accdaa19ed25f0")
SHA256 = dict(verify_py="22ce23b849fb03bf71cf73e2fa86865dd8edf9a898c63e684b7baa11ffdf7639",
              grader_py="f4fa06fed4950d6bd66a2fc40faaa67943327c2064e9d7ced1accdaa19ed25f0")
CASES = [
    # (answer block, reference, reward)
    # 1. lead-in: the disputed response and its neighbours
    ("Answer is 2. ", "2", 1), ("The answer is $16$", "16", 1), ("The answer is 255.", "255", 1), ("Therefore, the answer is $17$.", "17", 1),
    ("Answer: 2", "2", 1), ("The final answer is: 1/2", "\\frac{1}{2}", 1), ("Answer: 1,000", "1000", 1), ("The answer is -3", "-3", 1),
    ("Answer is 3.", "2", 0), ("The answer is not 5", "5", 0), ("The answer is 5 or 6", "5", 0), ("The answer is 3 and 4", "3", 0), ("The answer is 5, 5", "5", 0),
    ("The answer is -5", "5", 0), ("Answer is 12", "12_5", 0), ("Answer is (2,1)", "(1,2)", 0), ("Answer is 3.14", "\\pi", 0), ("The answer is 1,2", "12", 0),
    ("The answer is close to 5", "5", 0), ("The answer is 5 but maybe 6", "6", 0), ("Answer:", "5", 0), ("The answer is ", "5", 0), ("Answer is answer is 5", "5", 0),
    ("I think the answer is 5", "5", 0), ("5 is not the answer", "5", 0), ("No answer: 5", "5", 0), ("My answer is that there are 12 such integers.", "34", 0),
    # 2. factorial
    ("$8!$", "40,\\!320", 1), ("8!", "40320", 1), ("Answer: $8!$", "40,\\!320", 1), ("\\(5!\\)", "120", 1), ("The answer is 10!.", "3,628,800", 1), ("20!", "2432902008176640000", 1),
    ("8!", "40,\\!321", 0), ("7!", "40,\\!320", 0), ("9!/9", "40,\\!320", 0), ("8!!", "40,\\!320", 0), ("-8!", "40,\\!320", 0), ("8! or 7!", "40,\\!320", 0),
    ("21!", "51090942171709440000", 0), ("2!", "2_5", 0), ("0!", "0", 0), ("3!", "-6", 0), ("4!", "2_4", 0), ("4!", "(2,4)", 0), ("4!", "\frac{2}{4}", 0), ("4!", "2.4", 0),
    # literally 8!/9 = 4480, or an unclear derivation: the reviewed response itself stays rejected
    ("Answer: $8!$ by dividing by 9", "40,\\!320", 0), ("there are \\(8!\\)  ways", "40,\\!320", 0),
    # 3. unevaluated arithmetic and prose stay rejected
    ("16-3-4", "9", 0), ("9*8*7", "504", 0), ("42 - 12 = 30, so 30", "12", 0), ("$5$ is close", "5", 0),
    # 4. unchanged strict behaviour
    ("1/2", "\\frac{1}{2}", 1), ("$\\frac{29}{20}$", "\\frac{29}{20}", 1), ("2", "\\frac{1}{2}", 0), ("12,34", "1234", 0), ("[0,1)", "(0,1]", 0), 
    ("0.333", "\\frac13", 0), ("x=3 or x=4", "3", 0), ("\\boxed{4}", "4", 1), ("", "5", 0),
    # known leniency of the unchanged strict grader (units ignored; found in the Stage 0 audit): recorded, not endorsed
    ("6 feet", "6 \\text{ inches}", 1), ("3!", "\\frac{6}{1}", 1),
]
