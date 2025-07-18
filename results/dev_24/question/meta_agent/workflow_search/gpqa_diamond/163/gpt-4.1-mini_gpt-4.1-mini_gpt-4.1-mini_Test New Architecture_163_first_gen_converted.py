async def forward_163(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given quantitative data and relevant astrophysical relations from the query, "
        "including orbital periods, radial velocity amplitudes, and assumptions about orbits and inclinations. "
        "Use the provided query context to identify key parameters and assumptions."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Using the extracted data from Sub-task 1, combine the data to express the mass ratios of the stars in each system "
        "using the ratio of their radial velocity amplitudes, and relate orbital periods to total system masses via Kepler's third law. "
        "Debate the reasoning and calculations to find the most consistent relations."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and correct expressions for mass ratios and total mass relations "
        "based on radial velocity amplitudes and orbital periods."
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

    cot_sc_instruction3 = (
        "Sub-task 3: Compute the total mass of each binary system by applying Kepler's third law and the mass ratio relations "
        "derived from radial velocity amplitudes, assuming circular orbits and edge-on inclination. "
        "Use the relations and data from previous subtasks."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and select the most consistent total mass values for system_1 and system_2."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Derive the factor by which system_1 is more massive than system_2 by taking the ratio of their total masses computed in the previous step. "
        "Review and reflect on the limitations and consistency of the computed factor."
    )
    critic_instruction4 = (
        "Please review and provide the limitations of the provided solutions for the mass factor ratio between system_1 and system_2."
    )
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'critic_instruction': critic_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context_desc': [
            "user query", "thinking of subtask 1", "answer of subtask 1", 
            "thinking of subtask 2", "answer of subtask 2", 
            "thinking of subtask 3", "answer of subtask 3"
        ]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
