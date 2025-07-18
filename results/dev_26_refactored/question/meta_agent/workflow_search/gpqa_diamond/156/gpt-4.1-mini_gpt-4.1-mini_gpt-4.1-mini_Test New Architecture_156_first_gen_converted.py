async def forward_156(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Extract and transform the initial information about the retrovirus outbreak, including viral genome type, diagnostic targets (viral genome vs. antibodies), and available molecular techniques, to prepare a clear knowledge base for downstream analysis."
    )
    final_decision_instruction1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for extracting and transforming initial information about the retrovirus outbreak."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': final_decision_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_reflect_instruction2 = (
        "Sub-task 2: Combine and integrate the different identification methods (DNA sequencing, cDNA sequencing, symptom-based identification, antibody detection) and diagnostic kit development approaches (PCR variants, ELISA) to understand their interrelations, advantages, and limitations in the context of a retrovirus."
    )
    critic_instruction2 = (
        "Please review and provide the limitations of provided solutions of different identification and diagnostic approaches for retrovirus detection."
    )
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'critic_instruction': critic_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.reflexion(
        subtask_id="subtask_2",
        reflect_desc=cot_reflect_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate and select the most appropriate diagnostic approach for quick and accurate detection of the retrovirus by applying criteria such as viral genome type, detection speed, specificity, and feasibility, based on the integrated information from previous subtasks."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Select the best diagnostic approach for quick and accurate retrovirus detection."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'context_desc': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
