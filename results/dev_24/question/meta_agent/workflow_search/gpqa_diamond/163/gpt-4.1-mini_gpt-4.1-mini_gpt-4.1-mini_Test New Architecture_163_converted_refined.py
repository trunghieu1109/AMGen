async def forward_163(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Explicitly extract and record all given quantitative data from the query, "
        "including the orbital periods (2 years for system_1 and 1 year for system_2) and the radial velocity amplitudes "
        "for both stars in each system (10 km/s and 5 km/s for system_1; 15 km/s and 10 km/s for system_2). "
        "Embed the feedback that failure to extract these numbers caused previous errors, and emphasize that these values must be clearly identified and stored for subsequent calculations."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Verify and confirm the completeness and correctness of the extracted numerical data from Subtask 1. "
        "This verification step is critical to prevent the previous failure where agents incorrectly concluded no quantitative data were provided. "
        "Ensure that all six key numbers (two periods and four RV amplitudes) are present and correctly interpreted before proceeding."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent and correct solutions for the extracted numerical data. "
        "Given all the above thinking and answers, find the most consistent and correct solutions for the numerical data extraction."
    )
    cot_sc_desc2 = {
        'instruction': cot_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.reflexion(
        subtask_id="subtask_2",
        reflect_desc=cot_sc_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Using the verified numerical data, derive the mass ratios of the two stars in each system from the inverse ratio of their radial velocity amplitudes (q = K_secondary / K_primary). "
        "Emphasize the assumption of circular orbits and edge-on inclination due to eclipses, as per the problem statement and feedback. "
        "This step must explicitly apply the physics relations rather than vague qualitative reasoning."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and finalize the mass ratios for both systems based on the debate among agents."
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
        "Sub-task 4: Apply Kepler's third law to relate the orbital periods to the total masses of each binary system. "
        "Use the periods extracted and verified earlier, combined with the mass ratios from Subtask 3, to compute the total mass of system_1 and system_2. "
        "Assume similar inclination angles for both systems to allow cancellation in the ratio. "
        "This step must perform concrete calculations rather than qualitative assumptions, addressing the previous failure to use given data."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Synthesize and finalize the total masses of system_1 and system_2 based on the debate among agents."
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

    cot_reflect_instruction5 = (
        "Sub-task 5: Calculate the factor by which system_1 is more massive than system_2 by taking the ratio of their total masses computed in Subtask 4. "
        "Reflect on the assumptions made (circular orbits, edge-on inclination) and confirm that the final answer aligns with the quantitative data and physics principles. "
        "This final synthesis step should include a Reflexion pattern to critically review the entire reasoning chain and ensure no prior mistakes are repeated."
    )
    critic_instruction5 = (
        "Please review and provide the limitations of provided solutions for this problem, ensuring the final mass ratio factor is consistent and justified."
    )
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'critic_instruction': critic_instruction5,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.reflexion(
        subtask_id="subtask_5",
        reflect_desc=cot_reflect_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
