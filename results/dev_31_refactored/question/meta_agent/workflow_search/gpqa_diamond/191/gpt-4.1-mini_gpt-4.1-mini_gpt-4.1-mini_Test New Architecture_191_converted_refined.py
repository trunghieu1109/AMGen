async def forward_191(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Extract and summarize all given physical parameters, geometric relationships, and vector definitions from the problem statement to establish a clear problem framework. "
        "Input content are results (both thinking and answer) from: none."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Apply Gauss's law and electrostatics principles to analyze induced charges on the cavity and conductor surfaces, explicitly demonstrating that the net external charge on the conductor is +q at its center. "
        "This subtask addresses the previous failure of incorrectly assuming electrostatic shielding cancels the external field and ensures correct physical understanding of the problem. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)

    loop_results_stage_1 = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_instruction_1_1 = (
            f"Iteration {iteration+1} - Sub-task 1: Using the results from induced charge analysis, formulate the initial expression for the electric field magnitude at point P outside the conductor, "
            f"considering the conductor as a point charge +q located at its center. This subtask avoids the previous error of zero field assumption and grounds the expression in correct physics. "
            f"Input content are results (both thinking and answer) from: stage_0.subtask_2, respectively."
        )
        cot_agent_desc_1_1 = {
            "instruction": cot_instruction_1_1,
            "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']] + loop_results_stage_1["stage_1.subtask_1"]["thinking"] + loop_results_stage_1["stage_1.subtask_1"]["answer"],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"] + ["thinking of previous iterations of stage_1.subtask_1"]*len(loop_results_stage_1["stage_1.subtask_1"]["thinking"]) + ["answer of previous iterations of stage_1.subtask_1"]*len(loop_results_stage_1["stage_1.subtask_1"]["answer"])
        }
        results_1_1, log_1_1 = await self.cot(
            subtask_id=f"stage_1.subtask_1_iter_{iteration+1}",
            cot_agent_desc=cot_agent_desc_1_1
        )
        logs.append(log_1_1)
        loop_results_stage_1["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results_stage_1["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])

        cot_agent_instruction_1_2 = (
            f"Iteration {iteration+1} - Sub-task 2: Refine the electric field expression by incorporating vector geometry, including the displacement vector s, distance vectors l and L, and the angle theta, "
            f"to clarify the dependence of the field on these parameters. This subtask ensures the expression matches the problem's geometric setup and addresses previous ambiguity in vector definitions. "
            f"Input content are results (both thinking and answer) from: stage_1.subtask_1, respectively."
        )
        cot_agent_desc_1_2 = {
            "instruction": cot_agent_instruction_1_2,
            "input": [taskInfo] + loop_results_stage_1["stage_1.subtask_1"]["thinking"] + loop_results_stage_1["stage_1.subtask_1"]["answer"],
            "temperature": 0.0,
            "context": ["user query"] + ["thinking of stage_1.subtask_1"]*len(loop_results_stage_1["stage_1.subtask_1"]["thinking"]) + ["answer of stage_1.subtask_1"]*len(loop_results_stage_1["stage_1.subtask_1"]["answer"])
        }
        results_1_2, log_1_2 = await self.answer_generate(
            subtask_id=f"stage_1.subtask_2_iter_{iteration+1}",
            cot_agent_desc=cot_agent_desc_1_2
        )
        logs.append(log_1_2)
        loop_results_stage_1["stage_1.subtask_2"]["thinking"].append(results_1_2["thinking"])
        loop_results_stage_1["stage_1.subtask_2"]["answer"].append(results_1_2["answer"])

    cot_instruction_2_1 = (
        "Sub-task 1: Evaluate the candidate electric field expressions derived in stage_1 subtasks, verify their physical consistency with boundary conditions and induced charge results, "
        "and select the correct formula for the electric field magnitude at point P. This subtask prevents propagation of earlier misconceptions by cross-validating expressions. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1 & stage_1.subtask_2, respectively."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo] + loop_results_stage_1["stage_1.subtask_1"]["thinking"] + loop_results_stage_1["stage_1.subtask_1"]["answer"] + loop_results_stage_1["stage_1.subtask_2"]["thinking"] + loop_results_stage_1["stage_1.subtask_2"]["answer"],
        "temperature": 0.0,
        "context": ["user query"] + ["thinking of stage_1.subtask_1"]*len(loop_results_stage_1["stage_1.subtask_1"]["thinking"]) + ["answer of stage_1.subtask_1"]*len(loop_results_stage_1["stage_1.subtask_1"]["answer"]) + ["thinking of stage_1.subtask_2"]*len(loop_results_stage_1["stage_1.subtask_2"]["thinking"]) + ["answer of stage_1.subtask_2"]*len(loop_results_stage_1["stage_1.subtask_2"]["answer"])
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    cot_sc_instruction_3_1 = (
        "Sub-task 1: Check the selected electric field expression against known boundary conditions, such as behavior at large distances and conductor neutrality, to confirm its validity. "
        "This subtask explicitly addresses the previous failure of ignoring induced charges and ensures the final expression is physically sound. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_1, respectively."
    )
    final_decision_instruction_3_1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for the electric field expression validation problem."
    )
    cot_sc_desc_3_1 = {
        "instruction": cot_sc_instruction_3_1,
        "final_decision_instruction": final_decision_instruction_3_1,
        "input": [taskInfo, results_2_1["thinking"], results_2_1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.sc_cot(
        subtask_id="stage_3.subtask_1",
        cot_agent_desc=cot_sc_desc_3_1,
        n_repeat=self.max_sc
    )
    logs.append(log_3_1)

    formatter_instruction_4_1 = (
        "Sub-task 1: Present the final, validated electric field magnitude expression at point P in a clear, concise format consistent with the problem's notation and multiple-choice options. "
        "Input content are results (both thinking and answer) from: stage_3.subtask_1, respectively."
    )
    formatter_desc_4_1 = {
        "instruction": formatter_instruction_4_1,
        "input": [taskInfo, results_3_1["thinking"], results_3_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"],
        "format": "short and concise, without explaination"
    }
    results_4_1, log_4_1 = await self.specific_format(
        subtask_id="stage_4.subtask_1",
        formatter_desc=formatter_desc_4_1
    )
    logs.append(log_4_1)

    final_answer = await self.make_final_answer(results_4_1["thinking"], results_4_1["answer"])
    return final_answer, logs
