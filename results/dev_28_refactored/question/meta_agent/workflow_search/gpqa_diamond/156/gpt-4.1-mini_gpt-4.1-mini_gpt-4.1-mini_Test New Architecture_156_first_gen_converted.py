async def forward_156(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and categorize all relevant information from the query, including virus type (retrovirus), "
        "diagnostic goals (quick detection), possible identification methods (DNA sequencing, cDNA sequencing, antibody detection, symptom-based), "
        "and diagnostic techniques (PCR variants, ELISA). Identify constraints and key entities."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Analyze relationships between identified components: viral genome type (RNA retrovirus requiring reverse transcription), "
        "implications for sequencing method (cDNA vs DNA), diagnostic target (viral nucleic acid vs antibodies), and evaluate pros and cons of each diagnostic approach in context of speed and accuracy."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for analyzing relationships between viral genome and diagnostic methods."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Construct intermediate representations by mapping viral genome characteristics to suitable molecular diagnostic methods, "
        "selecting the optimal sequencing approach (cDNA sequencing for retrovirus RNA genome), and determining the best amplification technique (real-time PCR) for quick and accurate detection."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent solution for mapping viral genome to diagnostic methods."
    )
    cot_sc_desc3 = {
        "instruction": cot_sc_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Derive the detailed design workflow of the molecular diagnostic kit, including sample collection, reverse transcription to cDNA, primer design based on cDNA sequence, real-time PCR assay development, and validation steps to ensure specificity and sensitivity."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Synthesize and choose the most consistent detailed design workflow for the diagnostic kit."
    )
    cot_sc_desc4 = {
        "instruction": cot_sc_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Validate the proposed diagnostic kit design against criteria such as rapid turnaround time, accuracy, feasibility in outbreak conditions, "
        "and appropriateness for retroviral detection. Compare with alternative approaches (e.g., antibody detection, nested PCR) to justify the final choice."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Provide a final validated and justified diagnostic kit design choice."
    )
    debate_desc5 = {
        "instruction": debate_instruction5,
        "final_decision_instruction": final_decision_instruction5,
        "input": [taskInfo, results4["thinking"], results4["answer"]],
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "temperature": 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5["thinking"], results5["answer"])
    return final_answer, logs
