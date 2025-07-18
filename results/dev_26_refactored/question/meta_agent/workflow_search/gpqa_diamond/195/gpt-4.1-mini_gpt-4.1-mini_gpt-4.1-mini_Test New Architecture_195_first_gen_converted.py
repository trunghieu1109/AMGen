async def forward_195(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction_stage0_sub1 = (
        "Sub-task 1: Extract and summarize all given information and parameters from the problem statement, "
        "including physical setup, variables, and candidate formulas."
    )
    cot_agent_desc_stage0_sub1 = {
        "instruction": cot_instruction_stage0_sub1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_stage0_sub1, log_stage0_sub1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_stage0_sub1
    )
    logs.append(log_stage0_sub1)

    cot_sc_instruction_stage1_sub1 = (
        "Sub-task 1: Analyze the relativistic harmonic oscillator dynamics, including the relativistic energy and momentum relations, "
        "to derive an expression for the maximum speed v_max in terms of m, k, A, and c, based on the output from stage_0.subtask_1."
    )
    final_decision_instruction_stage1_sub1 = (
        "Sub-task 1: Synthesize and choose the most consistent derivation for the maximum speed v_max expression."
    )
    cot_sc_desc_stage1_sub1 = {
        "instruction": cot_sc_instruction_stage1_sub1,
        "final_decision_instruction": final_decision_instruction_stage1_sub1,
        "input": [taskInfo, results_stage0_sub1.get('thinking', ''), results_stage0_sub1.get('answer', '')],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_stage1_sub1, log_stage1_sub1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc_stage1_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage1_sub1)

    cot_reflect_instruction_stage1_sub2 = (
        "Sub-task 2: Integrate the physical constraints such as energy conservation and relativistic speed limits "
        "to refine the derived expression for v_max and ensure physical validity, based on outputs from stage_1.subtask_1."
    )
    critic_instruction_stage1_sub2 = (
        "Please review and provide the limitations of provided solutions for the maximum speed v_max of the relativistic harmonic oscillator."
    )
    cot_reflect_desc_stage1_sub2 = {
        "instruction": cot_reflect_instruction_stage1_sub2,
        "critic_instruction": critic_instruction_stage1_sub2,
        "input": [
            taskInfo,
            results_stage0_sub1.get('thinking', ''), results_stage0_sub1.get('answer', ''),
            results_stage1_sub1.get('thinking', ''), results_stage1_sub1.get('answer', '')
        ],
        "temperature": 0.0,
        "context": [
            "user query",
            "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1",
            "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"
        ]
    }
    results_stage1_sub2, log_stage1_sub2 = await self.reflexion(
        subtask_id="stage_1.subtask_2",
        reflect_desc=cot_reflect_desc_stage1_sub2,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub2)

    debate_instruction_stage2_sub1 = (
        "Sub-task 1: Evaluate the four candidate formulas for v_max against the derived expression and physical constraints "
        "to identify which formula correctly represents the maximum speed of the relativistic harmonic oscillator, based on outputs from stage_1."
    )
    final_decision_instruction_stage2_sub1 = (
        "Sub-task 1: Select the correct formula for v_max among the candidates given the analysis and constraints."
    )
    debate_desc_stage2_sub1 = {
        "instruction": debate_instruction_stage2_sub1,
        "final_decision_instruction": final_decision_instruction_stage2_sub1,
        "input": [
            taskInfo,
            results_stage1_sub2.get('thinking', ''), results_stage1_sub2.get('answer', '')
        ],
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        "temperature": 0.5
    }
    results_stage2_sub1, log_stage2_sub1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_stage2_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_sub1)

    final_answer = await self.make_final_answer(
        results_stage2_sub1.get('thinking', ''),
        results_stage2_sub1.get('answer', '')
    )

    return final_answer, logs
