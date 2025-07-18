async def forward_156(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify the key biological and technical elements involved in the diagnostic design problem, "
        "including the nature of the retrovirus (RNA genome), possible molecular targets (viral RNA, proviral DNA, antibodies), "
        "and diagnostic methods (PCR variants, ELISA), based on the given choices and biological context. "
        "Embed feedback to avoid overcomplicating the solution by focusing on biological completeness without considering practical constraints."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context_desc': ["user query"]
    }
    results_stage1_sub1, log1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Explicitly evaluate and score each candidate diagnostic approach (choices 1–4) against critical design constraints: "
        "speed, assay complexity, sample processing requirements, cost, and suitability for rapid, point-of-care molecular diagnostic kit deployment. "
        "Challenge multi-assay or multi-stage workflows and penalize complexity and time-consuming methods, embedding feedback to prevent groupthink and ensure constraint-based elimination of suboptimal options."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and correct solutions for the diagnostic approach evaluation. "
        "Given all the above thinking and answers, find the most consistent and correct solutions for the diagnostic design problem."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results_stage1_sub1['thinking'], results_stage1_sub1['answer']],
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        'temperature': 0.5
    }
    results_stage1_sub2, log2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_reflect_instruction3 = (
        "Sub-task 1: Select the optimal virus identification method and corresponding molecular diagnostic assay that balances biological accuracy with rapid, simple, and feasible deployment in outbreak settings. "
        "Justify the choice explicitly in terms of assay turnaround time, sample type feasibility, and resource constraints, embedding feedback to avoid prioritizing biological completeness over practical diagnostic design."
    )
    critic_instruction3 = (
        "Please review and provide the limitations of provided solutions for selecting the optimal diagnostic approach. "
        "Ensure the solution balances biological accuracy and practical deployment constraints."
    )
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'critic_instruction': critic_instruction3,
        'input': [taskInfo, results_stage1_sub1['thinking'], results_stage1_sub1['answer'], results_stage1_sub2['thinking'], results_stage1_sub2['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_stage2_sub1, log3 = await self.reflexion(
        subtask_id="stage_2.subtask_1",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_instruction4 = (
        "Sub-task 2: Design a detailed workflow for the molecular diagnostic kit development based on the selected optimal strategy, "
        "including sample type, assay protocol (e.g., real-time RT-PCR targeting viral RNA), and validation steps to ensure sensitivity and specificity. "
        "Incorporate practical considerations such as kit simplicity, rapid turnaround, and point-of-care applicability, explicitly addressing previous failures to consider these constraints."
    )
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results_stage2_sub1['thinking'], results_stage2_sub1['answer'], results_stage1_sub1['thinking']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1", "thinking of stage_1.subtask_1"]
    }
    results_stage2_sub2, log4 = await self.cot(
        subtask_id="stage_2.subtask_2",
        cot_agent_desc=cot_agent_desc4
    )
    logs.append(log4)

    cot_reflect_instruction5 = (
        "Sub-task 1: Perform a final critical evaluation and reflexion on the proposed diagnostic kit design, "
        "verifying that it meets the requirements for quick, accurate, and practical retrovirus detection in outbreak conditions. "
        "Challenge any residual assumptions about complexity or feasibility, ensuring the solution aligns with the brief and expert feedback to avoid overcomplication and impracticality."
    )
    critic_instruction5 = (
        "Please review and provide limitations or potential improvements for the final diagnostic kit design, "
        "ensuring it is practical, rapid, and accurate for outbreak conditions."
    )
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'critic_instruction': critic_instruction5,
        'input': [taskInfo, results_stage2_sub2['thinking'], results_stage2_sub2['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"]
    }
    results_stage3_sub1, log5 = await self.reflexion(
        subtask_id="stage_3.subtask_1",
        reflect_desc=cot_reflect_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results_stage3_sub1['thinking'], results_stage3_sub1['answer'])
    return final_answer, logs
