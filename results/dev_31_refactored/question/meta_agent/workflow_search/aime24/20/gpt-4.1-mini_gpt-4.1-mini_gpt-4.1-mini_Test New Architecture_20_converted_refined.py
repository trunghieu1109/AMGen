async def forward_20(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Formulate and analyze the key equation (d1 + d0)^2 = d1 * b + d0 with digit constraints (1 <= d1 <= b-1, 0 <= d0 <= b-1) and interpret its implications for b-eautiful numbers. This includes clarifying the relationship between digits, base, and the perfect square condition, ensuring no ambiguity in the problem setup. Input: taskInfo"
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc1)
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, derive explicit bounds and properties on digits d1, d0 and base b to limit the search space for solutions, based on the equation and digit constraints. This subtask must produce concrete numeric bounds or inequalities to guide enumeration, addressing the need to reduce computational complexity and avoid exhaustive brute force over impractical ranges. Input: taskInfo, results1[thinking], results1[answer]"
    )
    cot_agent_desc2 = {
        "instruction": cot_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results2, log2 = await self.cot(subtask_id="stage_0.subtask_2", cot_agent_desc=cot_agent_desc2)
    logs.append(log2)

    code_instruction1 = (
        "Sub-task 1: For each base b starting from 2, enumerate all digit pairs (d1, d0) satisfying the equation (d1 + d0)^2 = d1 * b + d0 under the digit constraints and bounds derived previously. Generate the explicit list of b-eautiful numbers (n = d1 * b + d0) for each base. This enumeration must be concrete and exhaustive within the feasible search space to avoid unsupported assumptions. Input: taskInfo, results2[thinking], results2[answer]"
    )

    code = """
def enumerate_beautiful_numbers(bounds_info):
    results = {}
    max_base_to_check = 1000
    for b in range(2, max_base_to_check + 1):
        beautiful_numbers = []
        for d1 in range(1, b):
            for d0 in range(0, b):
                s = d1 + d0
                n = d1 * b + d0
                if s * s == n:
                    beautiful_numbers.append(n)
        results[b] = beautiful_numbers
    return results

results = enumerate_beautiful_numbers(None)
"""

    code_agent_desc1 = {
        "instruction": code_instruction1,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"],
        "entry_point": "enumerate_beautiful_numbers"
    }

    results3, log3 = await self.logic_code(subtask_id="stage_1.subtask_1", code_agent_desc=code_agent_desc1)
    logs.append(log3)

    code_instruction2 = (
        "Sub-task 2: Count and tabulate the number of b-eautiful numbers for each base b enumerated in the previous subtask. This subtask must produce a clear mapping from base b to the count of b-eautiful numbers, explicitly addressing the previous failure of missing enumeration data and unsupported final conclusions. Input: taskInfo, results3[thinking], results3[answer]"
    )

    code2 = """
def count_beautiful_numbers(enumeration_results):
    counts = {}
    for b, nums in enumeration_results.items():
        counts[b] = len(nums)
    return counts

counts = count_beautiful_numbers(results)
"""

    code_agent_desc2 = {
        "instruction": code_instruction2,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "entry_point": "count_beautiful_numbers"
    }

    results4, log4 = await self.logic_code(subtask_id="stage_1.subtask_2", code_agent_desc=code_agent_desc2)
    logs.append(log4)

    cot_instruction3 = (
        "Sub-task 1: Analyze the enumeration and counting results to identify the smallest base b >= 2 for which the count of b-eautiful numbers exceeds ten. This subtask must rely explicitly on the tabulated counts from stage_1.subtask_2 to avoid unsupported guesses and ensure the final answer is data-driven and justified. Input: taskInfo, results4[thinking], results4[answer]"
    )
    cot_agent_desc3 = {
        "instruction": cot_instruction3,
        "input": [taskInfo, results4["thinking"], results4["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results5, log5 = await self.cot(subtask_id="stage_2.subtask_1", cot_agent_desc=cot_agent_desc3)
    logs.append(log5)

    final_answer = await self.make_final_answer(results5["thinking"], results5["answer"])
    return final_answer, logs
