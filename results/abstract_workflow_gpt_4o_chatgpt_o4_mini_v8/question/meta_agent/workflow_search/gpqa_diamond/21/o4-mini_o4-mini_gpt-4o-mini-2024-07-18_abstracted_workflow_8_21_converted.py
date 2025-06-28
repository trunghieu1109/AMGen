async def forward_21(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []

    # Sub-task 1
    cot1 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    instr1 = 'Sub-task 1: Identify the relevant half-reactions for oxygen reduction in acidic and basic media and state their standard reduction potentials.'
    thinking1, answer1 = await cot1([taskInfo], instr1, is_sub_task=True)
    agents.append(f'CoT agent {cot1.id}, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Subtask 1 answer: ', sub_tasks[-1])

    # Sub-task 2
    cot2 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    instr2 = 'Sub-task 2: Based on the potentials from Sub-task 1, determine whether oxygen is a weaker or stronger oxidant in basic solution.'
    thinking2, answer2 = await cot2([taskInfo, thinking1, answer1], instr2, is_sub_task=True)
    agents.append(f'CoT agent {cot2.id}, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Subtask 2 answer: ', sub_tasks[-1])

    # Sub-task 3
    cot3 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    instr3 = 'Sub-task 3: Collect known kinetic characteristics of the oxygen reduction reaction in acidic medium, focusing on its electron-transfer kinetics and overpotential requirements.'
    thinking3, answer3 = await cot3([taskInfo], instr3, is_sub_task=True)
    agents.append(f'CoT agent {cot3.id}, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Subtask 3 answer: ', sub_tasks[-1])

    # Sub-task 4
    cot4 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    instr4 = 'Sub-task 4: Using the kinetic data from Sub-task 3, conclude whether oxygen reduction is faster or slower in acidic solution compared to basic solution.'
    thinking4, answer4 = await cot4([taskInfo, thinking3, answer3], instr4, is_sub_task=True)
    agents.append(f'CoT agent {cot4.id}, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Subtask 4 answer: ', sub_tasks[-1])

    # Sub-task 5
    cot5 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    instr5 = 'Sub-task 5: Combine the thermodynamic conclusion from Sub-task 2 and the kinetic conclusion from Sub-task 4 into the ordered pair description.'
    thinking5, answer5 = await cot5([taskInfo, thinking2, answer2, thinking4, answer4], instr5, is_sub_task=True)
    agents.append(f'CoT agent {cot5.id}, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print('Subtask 5 answer: ', sub_tasks[-1])

    # Sub-task 6
    cot6 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    instr6 = 'Sub-task 6: Match the ordered pair from Sub-task 5 against the provided multiple-choice options and identify the correct choice.'
    thinking6, answer6 = await cot6([taskInfo, thinking5, answer5], instr6, is_sub_task=True)
    agents.append(f'CoT agent {cot6.id}, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    print('Subtask 6 answer: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer