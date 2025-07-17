async def forward_156(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and integrate key information about the retrovirus, including its genetic material type, "
        "possible molecular targets (viral genome vs. antibodies), and diagnostic methods (sequencing, PCR variants, ELISA). "
        "Classify these elements based on their relevance to quick and accurate detection."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Derive the optimal virus identification strategy by analyzing the molecular nature of the retrovirus and "
        "selecting the appropriate sequencing method (DNA or cDNA) and diagnostic amplification technique (PCR, nested PCR, or real-time PCR)."
    )
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

    cot_sc_instruction3 = (
        "Sub-task 3: Derive the diagnostic kit design approach by deciding between direct viral genome detection methods and host antibody detection methods, "
        "considering the timing of infection and diagnostic window for quick and accurate results."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Evaluate and prioritize the proposed diagnostic approaches (PCR-based vs. ELISA-based, sequencing methods, and PCR variants) "
        "based on criteria such as speed, accuracy, sensitivity, specificity, and suitability for retrovirus detection in an outbreak scenario."
    )
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
