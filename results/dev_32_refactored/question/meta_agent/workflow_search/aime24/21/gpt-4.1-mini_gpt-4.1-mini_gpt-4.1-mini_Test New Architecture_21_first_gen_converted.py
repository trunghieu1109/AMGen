async def forward_21(self, taskInfo):
    logs = []

    # stage_0.subtask_1
    cot_instruction_0_1 = (
        "Stage 0, Sub-task 1: Analyze the properties of the regular dodecagon and characterize which sides and diagonals can form rectangle edges. "
        "Input content: taskInfo"
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.5,
        "final_decision_instruction": "Stage 0, Sub-task 1, Final Decision: Synthesize and choose the most consistent answer for properties of the dodecagon and rectangle edges.",
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1,
        n_repeat=self.max_sc
    )
    logs.append(log_0_1)

    # stage_0.subtask_2
    cot_instruction_0_2 = (
        "Stage 0, Sub-task 2: Determine the conditions under which four edges (sides or diagonals) form a rectangle inside the dodecagon. "
        "Input content: results (thinking and answer) from stage_0.subtask_1"
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.5,
        "final_decision_instruction": "Stage 0, Sub-task 2, Final Decision: Synthesize and choose the most consistent answer for rectangle formation conditions.",
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.sc_cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2,
        n_repeat=self.max_sc
    )
    logs.append(log_0_2)

    loop_results = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_2.subtask_1": {"thinking": [], "answer": []},
        "stage_3.subtask_1": {"thinking": [], "answer": []}
    }

    for iteration in range(3):
        # stage_1.subtask_1
        cot_instruction_1_1 = (
            "Stage 1, Sub-task 1: Enumerate sets of four edges (sides or diagonals) that could potentially form rectangles based on geometric constraints. "
            "Input content: results (thinking and answer) from stage_0.subtask_2"
        )
        cot_agent_desc_1_1 = {
            "instruction": cot_instruction_1_1,
            "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']],
            "temperature": 0.6,
            "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
        }
        results_1_1, log_1_1 = await self.cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_agent_desc_1_1
        )
        logs.append(log_1_1)
        loop_results["stage_1.subtask_1"]["thinking"].append(results_1_1['thinking'])
        loop_results["stage_1.subtask_1"]["answer"].append(results_1_1['answer'])

        # stage_2.subtask_1
        cot_instruction_2_1 = (
            "Stage 2, Sub-task 1: Combine and filter the enumerated edge sets to consolidate a list of plausible rectangle candidates. "
            "Input content: results (thinking and answer) from stage_1.subtask_1"
        )
        cot_agent_desc_2_1 = {
            "instruction": cot_instruction_2_1,
            "input": [taskInfo] + loop_results["stage_1.subtask_1"]["thinking"] + loop_results["stage_1.subtask_1"]["answer"],
            "temperature": 0.6,
            "context_desc": ["user query"]
        }
        results_2_1, log_2_1 = await self.cot(
            subtask_id="stage_2.subtask_1",
            cot_agent_desc=cot_agent_desc_2_1
        )
        logs.append(log_2_1)
        loop_results["stage_2.subtask_1"]["thinking"].append(results_2_1['thinking'])
        loop_results["stage_2.subtask_1"]["answer"].append(results_2_1['answer'])

        # stage_3.subtask_1
        cot_instruction_3_1 = (
            "Stage 3, Sub-task 1: Validate each candidate rectangle against the criteria of rectangle formation and polygon constraints, selecting only valid rectangles. "
            "Input content: results (thinking and answer) from stage_2.subtask_1"
        )
        cot_agent_desc_3_1 = {
            "instruction": cot_instruction_3_1,
            "input": [taskInfo] + loop_results["stage_2.subtask_1"]["thinking"] + loop_results["stage_2.subtask_1"]["answer"],
            "temperature": 0.6,
            "final_decision_instruction": "Stage 3, Sub-task 1, Final Decision: Synthesize and choose the most consistent valid rectangles.",
            "context_desc": ["user query"]
        }
        results_3_1, log_3_1 = await self.sc_cot(
            subtask_id="stage_3.subtask_1",
            cot_agent_desc=cot_agent_desc_3_1,
            n_repeat=self.max_sc
        )
        logs.append(log_3_1)
        loop_results["stage_3.subtask_1"]["thinking"].append(results_3_1['thinking'])
        loop_results["stage_3.subtask_1"]["answer"].append(results_3_1['answer'])

    # stage_4.subtask_1
    cot_instruction_4_1 = (
        "Stage 4, Sub-task 1: Classify the valid rectangles by type or symmetry and resolve any duplicates or ambiguities. "
        "Input content: results (thinking and answer) from stage_3.subtask_1 and stage_0.subtask_2"
    )
    cot_agent_desc_4_1 = {
        "instruction": cot_instruction_4_1,
        "input": [taskInfo] + loop_results["stage_3.subtask_1"]["thinking"] + loop_results["stage_3.subtask_1"]["answer"] + [results_0_2['thinking'], results_0_2['answer']],
        "temperature": 0.5,
        "final_decision_instruction": "Stage 4, Sub-task 1, Final Decision: Synthesize and choose the most consistent classification and final rectangle list.",
        "context_desc": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results_4_1, log_4_1 = await self.sc_cot(
        subtask_id="stage_4.subtask_1",
        cot_agent_desc=cot_agent_desc_4_1,
        n_repeat=self.max_sc
    )
    logs.append(log_4_1)

    # stage_5.subtask_1
    formatter_instruction_5_1 = (
        "Stage 5, Sub-task 1: Format the count of valid rectangles into a clear, concise final answer. "
        "Input content: results (thinking and answer) from stage_4.subtask_1"
    )
    formatter_desc_5_1 = {
        "instruction": formatter_instruction_5_1,
        "input": [taskInfo, results_4_1['thinking'], results_4_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_4.subtask_1", "answer of stage_4.subtask_1"],
        "format": "short and concise, without explanation"
    }
    results_5_1, log_5_1 = await self.specific_format(
        subtask_id="stage_5.subtask_1",
        formatter_desc=formatter_desc_5_1
    )
    logs.append(log_5_1)

    final_answer = await self.make_final_answer(results_5_1['thinking'], results_5_1['answer'])
    return final_answer, logs
