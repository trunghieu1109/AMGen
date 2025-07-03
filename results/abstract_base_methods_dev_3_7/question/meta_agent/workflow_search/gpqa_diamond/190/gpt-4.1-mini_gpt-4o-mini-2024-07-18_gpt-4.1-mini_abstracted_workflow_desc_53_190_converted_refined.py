async def forward_190(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_reflect_instruction1 = (
        "Subtask 1: Analyze the reaction of 3-(hydroxymethyl)-5-(prop-1-en-2-yl)cyclohexan-1-one with sodium hydride followed by benzyl bromide. "
        "Explicitly verify that sodium hydride deprotonates the hydroxyl oxygen to form an alkoxide nucleophile, and that the substitution forms a C–O bond (benzyl ether), not a C–C bond. "
        "Provide the structure of product 1 and validate the bond formation type before proceeding."
    )
    critic_instruction1 = (
        "Please review the proposed product 1 structure, focusing on the nucleophile identity and bond formation type. "
        "Provide feedback on any mechanistic or structural errors and suggest corrections."
    )
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1,
        'input': [taskInfo],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ['user query']
    }
    critic_desc1 = {
        'instruction': critic_instruction1,
        'output': ['feedback', 'correct'],
        'temperature': 0.0
    }
    results1 = await self.reflexion(
        subtask_id="subtask_1",
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining product 1 structure, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results1['critic_agent'].id}, round {i}, feedback: {results1['list_feedback'][i].content}; correctness: {results1['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, round {i}, refining product 1 structure, thinking: {results1['list_thinking'][i + 1].content}; answer: {results1['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction2 = (
        "Subtask 2: Based on the validated structure of product 1, analyze the reaction with p-toluenesulfonyl hydrazide in catalytic HCl to form product 2. "
        "Cross-validate the hydrazone formation step with the verified product 1 structure and pause to verify before proceeding."
    )
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_instruction=cot_sc_instruction2,
        input_list=[taskInfo, results1['thinking'], results1['answer']],
        output_fields=['thinking', 'answer'],
        temperature=0.5,
        context=['user query', results1['thinking'], results1['answer']],
        n_repeat=self.max_sc
    )
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    for idx, key in enumerate(results2['list_thinking']):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, cross-validating hydrazone formation, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    logs.append(results2['subtask_desc'])

    debate_instruction3 = (
        "Subtask 3: Analyze the reaction of product 2 with two equivalents of n-butyllithium at low temperature followed by aqueous ammonium chloride, i.e., the Shapiro reaction. "
        "Explicitly outline the mechanism including diazo intermediate formation, elimination of N2 and tosyl group, and alkene formation. "
        "Debate possible product structures, argue for or against each, and clarify the reaction sequence to determine product 3."
    )
    final_decision_instruction3 = "Subtask 3: Make final decision on the correct structure of product 3 based on the debate."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ['user query', results2['thinking'], results2['answer']],
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5
    }
    final_decision_desc3 = {
        'instruction': final_decision_instruction3,
        'output': ['thinking', 'answer'],
        'temperature': 0.0
    }
    results3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        final_decision_desc=final_decision_desc3,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results3['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, debating Shapiro reaction product, thinking: {results3['list_thinking'][round][idx].content}; answer: {results3['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding product 3 structure, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    cot_instruction4 = (
        "Subtask 4: Analyze the hydrogenation of product 3 with Pd/C under hydrogen atmosphere. "
        "List all reducible moieties such as alkenes and benzyl ethers, discuss their fate under these conditions, and confirm the structure of product 4."
    )
    results4 = await self.cot(
        subtask_id="subtask_4",
        cot_instruction=cot_instruction4,
        input_list=[taskInfo, results3['thinking'], results3['answer']],
        output_fields=['thinking', 'answer'],
        temperature=0.0,
        context=['user query', results3['thinking'], results3['answer']]
    )
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    agents.append(f"CoT agent {results4['cot_agent'].id}, analyzing hydrogenation, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    debate_instruction5 = (
        "Subtask 5: Compare the deduced structure of product 4 with the given multiple-choice options (A, B, C, D). "
        "Justify why each choice matches or does not match the deduced structure, then debate and vote on the best answer choice."
    )
    final_decision_instruction5 = "Subtask 5: Make final decision on the correct answer choice for product 4's structure."
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ['user query', results4['thinking'], results4['answer']],
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5
    }
    final_decision_desc5 = {
        'instruction': final_decision_instruction5,
        'output': ['thinking', 'answer'],
        'temperature': 0.0
    }
    results5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        final_decision_desc=final_decision_desc5,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results5['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, debating answer choice, thinking: {results5['list_thinking'][round][idx].content}; answer: {results5['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, calculating final answer, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
