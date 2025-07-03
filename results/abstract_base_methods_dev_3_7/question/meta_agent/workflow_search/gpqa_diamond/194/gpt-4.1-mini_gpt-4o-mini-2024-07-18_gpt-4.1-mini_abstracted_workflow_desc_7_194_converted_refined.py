async def forward_194(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Calculate the orbital radius 'a' of the first planet using Kepler's third law: "
        "P^2 = (4 * pi^2 * a^3) / (G * M_star). Assume stellar mass M_star = 1.5 solar masses (M_sun = 1.989e30 kg), "
        "orbital period P = 3 days (converted to seconds), and G = 6.67430e-11 m^3 kg^-1 s^-2. "
        "Show all unit conversions and numerical substitutions step-by-step, then solve for 'a' in meters and convert to solar radii."
    )
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_instruction=cot_instruction_1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context="user query"
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, calculating orbital radius of first planet, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction_2 = (
        "Sub-task 2: Using the orbital radius 'a' from Sub-task 1, calculate the orbital inclination 'i' of the first planet. "
        "Given the transit impact parameter b = 0.2, star radius R_star = 1.5 solar radii, compute cos(i) = b * R_star / a numerically, "
        "then calculate i in degrees by arccos. Show all steps explicitly."
    )
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_instruction=cot_instruction_2,
        input_list=[taskInfo, results1['thinking'], results1['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user query", results1['thinking'], results1['answer']]
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, calculating orbital inclination, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    debate_instruction_3 = (
        "Sub-task 3: Derive the geometric inequalities for a planet to exhibit both transit and occultation events. "
        "Express the transit condition as impact parameter b ≤ 1 + (R_p / R_star) and occultation condition as b ≤ 1 - (R_p / R_star). "
        "Relate these inequalities to the orbital radius 'a', planet radius R_p, star radius R_star, and orbital inclination 'i'. "
        "Use debate among agents to challenge and verify these inequalities and their implications for the second planet's orbit."
    )
    debate_desc_3 = {
        "instruction": debate_instruction_3,
        "context": ["user query", results2['thinking'], results2['answer']],
        "input": [taskInfo, results2['thinking'], results2['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    results3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc_3,
        final_decision_desc=None,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results3['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, deriving geometric inequalities, thinking: {results3['list_thinking'][round][idx].content}; answer: {results3['list_answer'][round][idx].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    cot_sc_instruction_4 = (
        "Sub-task 4: Using the inequalities from Sub-task 3, numerically solve for the maximum semi-major axis 'a_max' of the second planet that satisfies both transit and occultation conditions. "
        "Then, apply Kepler's third law explicitly with M_star = 1.5 solar masses to calculate the corresponding maximum orbital period P_max in days. "
        "Include all intermediate steps, unit conversions, and cross-verification of results."
    )
    results4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_sc_instruction=cot_sc_instruction_4,
        input_list=[taskInfo, results3['thinking'], results3['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", results3['thinking'], results3['answer']],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"SC_CoT agent {results4['cot_agent'][idx].id}, solving max orbital radius and period, thinking: {results4['list_thinking'][idx]}; answer: {results4['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    cot_reflect_instruction_5 = (
        "Sub-task 5: Convert the calculated maximum orbital period from Sub-task 4 to the closest multiple-choice letter option (A: ~7.5, B: ~33.5, C: ~37.5, D: ~12.5 days). "
        "Verify that the numeric answer aligns with one of the options and output only the letter choice."
    )
    critic_instruction_5 = "Please review the conversion and selection of the multiple-choice letter and provide any limitations or corrections needed."
    cot_reflect_desc_5 = {
        'instruction': cot_reflect_instruction_5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", results4['thinking'], results4['answer']]
    }
    critic_desc_5 = {
        'instruction': critic_instruction_5,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results5 = await self.reflexion(
        subtask_id="subtask_5",
        cot_reflect_desc=cot_reflect_desc_5,
        critic_desc=critic_desc_5,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, converting numeric period to letter choice, thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results5['critic_agent'].id}, providing feedback, thinking: {results5['list_feedback'][i].content}; answer: {results5['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining letter choice, thinking: {results5['list_thinking'][i + 1].content}; answer: {results5['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
