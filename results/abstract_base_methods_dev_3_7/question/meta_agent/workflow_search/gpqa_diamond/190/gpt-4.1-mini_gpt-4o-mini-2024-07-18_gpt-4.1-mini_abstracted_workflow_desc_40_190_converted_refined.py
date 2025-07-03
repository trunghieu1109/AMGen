async def forward_190(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_reflect_instruction_1 = (
        "Sub-task 1: Analyze the starting material 3-(hydroxymethyl)-5-(prop-1-en-2-yl)cyclohexan-1-one and the reaction with sodium hydride and benzyl bromide. "
        "Clarify that sodium hydride deprotonates the hydroxyl group and benzyl bromide alkylates the oxygen, forming a benzyloxymethyl ether (R–O–CH2Ph). "
        "Verify explicitly that the O–CH2Ph linkage is retained and the hydroxymethyl substituent is converted to a benzyloxymethyl ether, not replaced."
    )
    critic_instruction_1 = (
        "Please review the alkylation product structure and reasoning for any chemical inaccuracies or misunderstandings, especially regarding the ether formation and substituent retention."
    )
    cot_reflect_desc_1 = {
        'instruction': cot_reflect_instruction_1,
        'input': [taskInfo],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ['user query']
    }
    critic_desc_1 = {
        'instruction': critic_instruction_1,
        'output': ['feedback', 'correct'],
        'temperature': 0.0
    }
    results1 = await self.reflexion(
        subtask_id='subtask_1',
        cot_reflect_desc=cot_reflect_desc_1,
        critic_desc=critic_desc_1,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, analyzing alkylation, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback round {i}, thinking: {results1['list_feedback'][i].content}; answer: {results1['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining alkylation analysis round {i}, thinking: {results1['list_thinking'][i+1].content}; answer: {results1['list_answer'][i+1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction_2 = (
        "Sub-task 2: Based on the corrected product 1 structure (benzyloxymethyl ether), analyze the reaction with p-toluenesulfonyl hydrazide and catalytic HCl to form product 2. "
        "Confirm the presence and position of the ketone carbonyl before tosylhydrazone formation. "
        "Include the exact substituent (benzyloxymethyl) in the product 2 name to reflect the correct structure."
    )
    results2 = await self.cot(
        subtask_id='subtask_2',
        cot_instruction=cot_instruction_2,
        input_list=[taskInfo, results1['thinking'], results1['answer']],
        output_fields=['thinking', 'answer'],
        temperature=0.0,
        context=['user query', 'thinking of subtask 1', 'answer of subtask 1']
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, analyzing tosylhydrazone formation, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_sc_instruction_3 = (
        "Sub-task 3: Based on product 2, analyze the treatment with n-butyllithium at low temperature followed by aqueous ammonium chloride (Shapiro reaction). "
        "Outline two or three possible alkene products formed from the tosylhydrazone, providing mechanistic justification for each. "
        "Select the most chemically plausible product consistent with Shapiro reaction rules."
    )
    results3 = await self.sc_cot(
        subtask_id='subtask_3',
        cot_sc_instruction=cot_sc_instruction_3,
        input_list=[taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        output_fields=['thinking', 'answer'],
        temperature=0.5,
        context=['user query', 'thinking of subtask 1', 'answer of subtask 1', 'thinking of subtask 2', 'answer of subtask 2'],
        n_repeat=self.max_sc
    )
    for idx, _ in enumerate(results3['list_thinking']):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, analyzing Shapiro reaction, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    debate_instruction_4 = (
        "Sub-task 4: Analyze the catalytic hydrogenation of product 3 with Pd/C under hydrogen atmosphere. "
        "Discuss not only reduction of double bonds but also possible cleavage of benzyl ether protecting groups. "
        "Encourage agents to debate the stability of substituents and reconcile different plausible outcomes to deduce product 4 structure."
    )
    final_decision_instruction_4 = "Sub-task 4: Make final decision on the structure of product 4 after hydrogenation considering all side reactions."
    debate_desc_4 = {
        'instruction': debate_instruction_4,
        'context': ['user query', results3['thinking'], results3['answer']],
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.5
    }
    final_decision_desc_4 = {
        'instruction': final_decision_instruction_4,
        'output': ['thinking', 'answer'],
        'temperature': 0.0
    }
    results4 = await self.debate(
        subtask_id='subtask_4',
        debate_desc=debate_desc_4,
        final_decision_desc=final_decision_desc_4,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results4['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, analyzing hydrogenation, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding product 4 structure, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    cot_reflect_instruction_5 = (
        "Sub-task 5: Based on the deduced product 4 structure, list the structure explicitly and evaluate each multiple-choice option in turn for an exact match. "
        "If none of the options match, reconsider previous steps or explicitly state that no option fits the deduced structure. "
        "Provide a final reasoned selection of the correct answer choice."
    )
    critic_instruction_5 = (
        "Please review the final answer selection process, checking for consistency with the deduced product 4 structure and the given choices. "
        "Suggest corrections or reconsiderations if no choice matches."
    )
    cot_reflect_desc_5 = {
        'instruction': cot_reflect_instruction_5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 4', 'answer of subtask 4']
    }
    critic_desc_5 = {
        'instruction': critic_instruction_5,
        'output': ['feedback', 'correct'],
        'temperature': 0.0
    }
    results5 = await self.reflexion(
        subtask_id='subtask_5',
        cot_reflect_desc=cot_reflect_desc_5,
        critic_desc=critic_desc_5,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, final answer evaluation, thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {results5['critic_agent'].id}, feedback round {i}, thinking: {results5['list_feedback'][i].content}; answer: {results5['list_correct'][i].content}")
        agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining final answer round {i}, thinking: {results5['list_thinking'][i+1].content}; answer: {results5['list_answer'][i+1].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs