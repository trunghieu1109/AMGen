async def forward_157(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Extract and summarize the key molecular and genetic information from the query, "
        "including the transcription factor activation mechanism, mutation X and Y characteristics, and their domain locations."
    )
    final_decision_instruction1 = (
        "Sub-task 1: Synthesize and choose the most consistent summary of the key molecular and genetic information "
        "from the query."
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

    debate_instruction2 = (
        "Sub-task 2: Classify the types of mutations described in the query (recessive loss-of-function vs dominant-negative) "
        "and analyze their typical molecular consequences, especially focusing on the dominant-negative mutation Y in the dimerization domain."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Based on the classification and analysis, provide a detailed explanation of the molecular consequences "
        "of mutation Y as a dominant-negative mutation."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"],
        'temperature': 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Analyze the molecular mechanisms by which a dominant-negative mutation in the dimerization domain "
        "can affect protein function, including possible effects on dimerization, protein stability, aggregation, degradation, and functional outcome."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide a comprehensive mechanistic explanation of how mutation Y exerts its dominant-negative effect."
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

    debate_instruction4 = (
        "Sub-task 4: Evaluate each of the four provided molecular phenotype options in the context of the dominant-negative mutation's mechanism "
        "and select the most likely molecular phenotype observed with mutation Y."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Choose the best molecular phenotype option that explains the dominant-negative effect of mutation Y."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3"],
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
