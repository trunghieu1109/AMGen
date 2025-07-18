async def forward_157(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Extract and summarize the key biological information about the transcription factor subunit, "
        "including activation mechanism, phosphorylation sites, dimerization, nuclear translocation, and gene transcription role, "
        "with context from the provided query."
    )
    cot_sc_final_decision1 = (
        "Sub-task 1: Synthesize and choose the most consistent summary of the transcription factor subunit's biological information."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': cot_sc_final_decision1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    thinking1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Extract and summarize the characteristics of mutations X and Y, including their location (transactivation vs dimerization domain), "
        "mutation type (missense), inheritance pattern (recessive vs dominant-negative), and functional consequences, "
        "based on the output from Sub-task 1 and the provided query."
    )
    cot_sc_final_decision2 = (
        "Sub-task 2: Synthesize and choose the most consistent summary of mutation characteristics."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': cot_sc_final_decision2,
        'input': [taskInfo, thinking1],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1"]
    }
    thinking2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Analyze the molecular mechanism of dominant-negative mutations in dimerization domains, "
        "focusing on how mutant proteins interfere with wild-type protein function, including possible effects like loss of dimerization, aggregation, degradation, or conformational changes, "
        "based on the summaries from Sub-task 2."
    )
    debate_final_decision3 = (
        "Sub-task 3: Provide a reasoned analysis of dominant-negative mutation mechanisms in the dimerization domain."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': debate_final_decision3,
        'input': [taskInfo, thinking2],
        'context_desc': ["user query", "thinking of subtask 2"],
        'temperature': 0.5
    }
    thinking3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Evaluate each provided molecular phenotype option in the context of dominant-negative mutation Y, "
        "comparing them against typical dominant-negative effects and the biological context of the transcription factor, "
        "based on the analysis from Sub-task 3 and biological information from Sub-task 1."
    )
    debate_final_decision4 = (
        "Sub-task 4: Provide a reasoned evaluation of the molecular phenotype options for mutation Y."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'final_decision_instruction': debate_final_decision4,
        'input': [taskInfo, thinking3, thinking1],
        'context_desc': ["user query", "thinking of subtask 3", "thinking of subtask 1"],
        'temperature': 0.5
    }
    thinking4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_reflect_instruction5 = (
        "Sub-task 5: Select the most likely molecular phenotype observed in the presence of mutation Y based on the analysis, "
        "providing a reasoned justification for the choice, considering all previous subtasks' outputs."
    )
    critic_instruction5 = (
        "Please review and provide the limitations of the provided solutions and justify the final choice of molecular phenotype for mutation Y."
    )
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'critic_instruction': critic_instruction5,
        'input': [taskInfo, thinking1, thinking2, thinking3, thinking4],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 1", "thinking of subtask 2", "thinking of subtask 3", "thinking of subtask 4"]
    }
    thinking5, log5 = await self.reflexion(
        subtask_id="subtask_5",
        reflect_desc=cot_reflect_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(thinking5, None)
    return final_answer, logs
