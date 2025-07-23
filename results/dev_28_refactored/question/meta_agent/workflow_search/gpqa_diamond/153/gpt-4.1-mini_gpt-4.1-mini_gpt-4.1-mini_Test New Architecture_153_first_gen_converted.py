async def forward_153(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and categorize all relevant spectral data and molecular information from the input query, "
        "including mass spectrometry peaks, IR absorption bands, and 1H NMR signals, to create a structured summary of the compound's characteristics."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Analyze the extracted spectral data from Sub-task 1 to identify functional groups, substitution patterns, "
        "and isotopic signatures; derive intermediate structural insights such as presence of chlorine, carboxylic acid group, "
        "para-substitution on the aromatic ring, and molecular weight confirmation."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the spectral data analysis and structural insights."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1['thinking'], results1['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate the candidate structures (4-chlorobenzoic acid, 2-chlorobenzoic acid, 3-chloro-2-hydroxybenzaldehyde, phenyl chloroformate) "
        "against the derived spectral and structural criteria from Sub-task 2 to select the best fitting compound that matches the molecular weight, isotope pattern, IR and NMR data."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Select the most appropriate compound based on spectral data and structural analysis."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2['thinking'], results2['answer']],
        "context_desc": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
