async def forward_156(self, taskInfo):
    logs = []

    cot_instruction_stage0 = (
        "Sub-task 1: Analyze the outbreak context and identify key elements such as virus type (retrovirus), diagnostic goals, and candidate detection methods from the query. "
        "Input content: taskInfo containing question and choices."
    )
    cot_agent_desc_stage0 = {
        "instruction": cot_instruction_stage0,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_stage0, log_stage0 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_stage0
    )
    logs.append(log_stage0)

    cot_instruction_stage1 = (
        "Sub-task 1: Map virus biology (RNA retrovirus) to appropriate identification methods (cDNA sequencing) and diagnostic techniques (real-time PCR), "
        "generating structured intermediate steps for kit design. Input content: results (thinking and answer) from stage_0.subtask_1 and taskInfo."
    )
    cot_agent_desc_stage1 = {
        "instruction": cot_instruction_stage1,
        "input": [taskInfo, results_stage0['thinking'], results_stage0['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_stage1, log_stage1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_stage1
    )
    logs.append(log_stage1)

    loop_results = {
        "stage_2.subtask_1": {"thinking": [], "answer": []},
        "stage_4.subtask_1": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_sc_instruction_stage2 = (
            "Sub-task 1: Evaluate candidate diagnostic approaches (DNA sequencing + PCR, IgG ELISA, symptom-based nested PCR, cDNA sequencing + real-time PCR) "
            "against criteria of speed, accuracy, and retrovirus biology. Input content: results from stage_1.subtask_1 and all previous iterations of stage_4.subtask_1."
        )
        final_decision_instruction_stage2 = (
            "Sub-task 1: Synthesize and choose the most consistent answer for evaluating candidate diagnostic approaches."
        )
        cot_sc_desc_stage2 = {
            "instruction": cot_sc_instruction_stage2,
            "final_decision_instruction": final_decision_instruction_stage2,
            "input": [taskInfo, results_stage1['thinking'], results_stage1['answer']] + loop_results["stage_4.subtask_1"]["thinking"] + loop_results["stage_4.subtask_1"]["answer"],
            "temperature": 0.5,
            "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"] + ["thinking of stage_4.subtask_1"]*len(loop_results["stage_4.subtask_1"]["thinking"]) + ["answer of stage_4.subtask_1"]*len(loop_results["stage_4.subtask_1"]["answer"])
        }
        results_stage2, log_stage2 = await self.sc_cot(
            subtask_id="stage_2.subtask_1",
            cot_agent_desc=cot_sc_desc_stage2,
            n_repeat=self.max_sc
        )
        logs.append(log_stage2)

        cot_reflect_instruction_stage4 = (
            "Sub-task 1: Assess the validity and suitability of the selected diagnostic method for retrovirus detection, providing feedback for refinement. "
            "Input content: results from stage_2.subtask_1."
        )
        critic_instruction_stage4 = (
            "Please review and provide the limitations of provided solutions of the selected diagnostic method for retrovirus detection."
        )
        cot_reflect_desc_stage4 = {
            "instruction": cot_reflect_instruction_stage4,
            "critic_instruction": critic_instruction_stage4,
            "input": [taskInfo, results_stage2['thinking'], results_stage2['answer']],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
        }
        results_stage4, log_stage4 = await self.reflexion(
            subtask_id="stage_4.subtask_1",
            reflect_desc=cot_reflect_desc_stage4,
            n_repeat=self.max_round
        )
        logs.append(log_stage4)

        loop_results["stage_2.subtask_1"]["thinking"].append(results_stage2['thinking'])
        loop_results["stage_2.subtask_1"]["answer"].append(results_stage2['answer'])
        loop_results["stage_4.subtask_1"]["thinking"].append(results_stage4['thinking'])
        loop_results["stage_4.subtask_1"]["answer"].append(results_stage4['answer'])

    final_answer = await self.make_final_answer(loop_results["stage_4.subtask_1"]["thinking"][-1], loop_results["stage_4.subtask_1"]["answer"][-1])
    return final_answer, logs
