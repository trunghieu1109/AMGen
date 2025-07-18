async def forward_156(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = "Sub-task 1: Identify and characterize the essential biological and molecular features of the retrovirus relevant for diagnostic design, including genome type (RNA/DNA), presence of viral RNA or DNA, and suitable molecular targets for detection, based on the query provided."
    debate_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=debate_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = "Sub-task 2: Analyze and classify potential diagnostic methods (DNA sequencing, cDNA sequencing, antibody detection, symptom-based inference) based on their suitability for quick and accurate detection of the retrovirus, considering molecular targets and assay types (PCR, nested PCR, real-time PCR, ELISA), using outputs from Sub-task 1."
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_reflect_instruction3 = "Sub-task 3: Assess the impact of each diagnostic approach on speed, accuracy, feasibility, and specificity, including technical considerations such as the need for reverse transcription in RNA viruses and the reliability of antibody detection versus direct viral detection, based on outputs from Sub-task 1 and Sub-task 2."
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = "Sub-task 4: Select the optimal molecular diagnostic approach and design strategy for the kit, prioritizing quick and accurate detection of the retrovirus, and justify the choice based on the analysis of viral features and diagnostic method assessments from previous subtasks."
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"],
        'input': [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
