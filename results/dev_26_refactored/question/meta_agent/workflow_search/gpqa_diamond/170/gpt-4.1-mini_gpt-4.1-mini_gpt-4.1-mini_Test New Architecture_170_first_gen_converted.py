async def forward_170(self, taskInfo):
    logs = []

    cot_instruction_stage0_sub1 = (
        "Sub-task 1: Extract and summarize the chemical structures and substituents of the six substances, "
        "and clarify the reaction conditions and assumptions (monobromo derivative, electrophilic substitution with excess bromine)."
    )
    cot_agent_desc_stage0_sub1 = {
        "instruction": cot_instruction_stage0_sub1,
        "input": [taskInfo],
        "temperature": 0.5,
        "context": ["user query"]
    }
    results_stage0_sub1, log_stage0_sub1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_stage0_sub1,
        n_repeat=self.max_sc
    )
    logs.append(log_stage0_sub1)

    debate_instruction_stage1_sub1 = (
        "Sub-task 1: Analyze the electronic effects (activating/deactivating) and directing effects (ortho/para vs meta) "
        "of each substituent on the benzene ring to predict regioselectivity in bromination."
    )
    final_decision_instruction_stage1_sub1 = (
        "Sub-task 1: Synthesize and choose the most consistent analysis of electronic and directing effects for the substituents."
    )
    debate_desc_stage1_sub1 = {
        "instruction": debate_instruction_stage1_sub1,
        "final_decision_instruction": final_decision_instruction_stage1_sub1,
        "input": [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer']],
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
        "temperature": 0.5
    }
    results_stage1_sub1, log_stage1_sub1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc_stage1_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage1_sub1)

    cot_sc_instruction_stage1_sub2 = (
        "Sub-task 2: Evaluate steric and electronic factors influencing the relative yields of para- versus ortho-isomers for each substituent, "
        "estimating the expected weight fraction of the para-isomer, based on outputs from stage_0.subtask_1 and stage_1.subtask_1."
    )
    final_decision_instruction_stage1_sub2 = (
        "Sub-task 2: Synthesize and choose the most consistent evaluation of steric and electronic factors affecting para-isomer yields."
    )
    cot_sc_desc_stage1_sub2 = {
        "instruction": cot_sc_instruction_stage1_sub2,
        "final_decision_instruction": final_decision_instruction_stage1_sub2,
        "input": [taskInfo, results_stage0_sub1['thinking'], results_stage0_sub1['answer'], results_stage1_sub1['thinking'], results_stage1_sub1['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_stage1_sub2, log_stage1_sub2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_sc_desc_stage1_sub2,
        n_repeat=self.max_sc
    )
    logs.append(log_stage1_sub2)

    debate_instruction_stage2_sub1 = (
        "Sub-task 1: Integrate the substituent effects and regioselectivity analysis to rank the six substances in order of increasing weight fraction of the para-bromo isomer yield, "
        "based on outputs from stage_1.subtask_1 and stage_1.subtask_2."
    )
    final_decision_instruction_stage2_sub1 = (
        "Sub-task 1: Provide a consistent and justified ranking of the substances by increasing para-isomer weight fraction."
    )
    debate_desc_stage2_sub1 = {
        "instruction": debate_instruction_stage2_sub1,
        "final_decision_instruction": final_decision_instruction_stage2_sub1,
        "input": [taskInfo, results_stage1_sub1['thinking'], results_stage1_sub1['answer'], results_stage1_sub2['thinking'], results_stage1_sub2['answer']],
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        "temperature": 0.5
    }
    results_stage2_sub1, log_stage2_sub1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc_stage2_sub1,
        n_repeat=self.max_round
    )
    logs.append(log_stage2_sub1)

    cot_sc_instruction_stage2_sub2 = (
        "Sub-task 2: Compare the derived ranking with the provided multiple-choice options and select the correct order, "
        "based on output from stage_2.subtask_1."
    )
    final_decision_instruction_stage2_sub2 = (
        "Sub-task 2: Choose the correct multiple-choice answer that matches the ranking."
    )
    cot_sc_desc_stage2_sub2 = {
        "instruction": cot_sc_instruction_stage2_sub2,
        "final_decision_instruction": final_decision_instruction_stage2_sub2,
        "input": [taskInfo, results_stage2_sub1['thinking'], results_stage2_sub1['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_stage2_sub2, log_stage2_sub2 = await self.sc_cot(
        subtask_id="stage_2.subtask_2",
        cot_agent_desc=cot_sc_desc_stage2_sub2,
        n_repeat=self.max_sc
    )
    logs.append(log_stage2_sub2)

    final_answer = await self.make_final_answer(results_stage2_sub2['thinking'], results_stage2_sub2['answer'])
    return final_answer, logs
