async def forward_22(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    cot_instruction1 = 'Sub-task 1: Parse the IUPAC name ((2,2-dimethylbut-3-en-1-yl)oxy)benzene to draw or represent its molecular structure with all atoms, bonds, functional groups, and stereochemical features clear.'
    cot_agent1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, parsing IUPAC name, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print('Subtask 1 answer: ', sub_tasks[-1])
    cot_sc_instruction2 = 'Sub-task 2: From the drawn structure, identify and classify reactive moieties0alkene and aryl ether link0and note any site that can undergo protonation or carbocation formation.'
    N = self.max_sc
    sc_agents2 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers2 = []
    thinkingmapping2 = {}
    answermapping2 = {}
    for i in range(N):
        thinking2_i, answer2_i = await sc_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents2[i].id}, identifying reactive moieties, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers2.append(answer2_i.content)
        thinkingmapping2[answer2_i.content] = thinking2_i
        answermapping2[answer2_i.content] = answer2_i
    most_common2 = Counter(possible_answers2).most_common(1)[0][0]
    thinking2 = thinkingmapping2[most_common2]
    answer2 = answermapping2[most_common2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print('Subtask 2 answer: ', sub_tasks[-1])
    cot_reflect_instruction3 = 'Sub-task 3: Outline the mechanistic sequence when HBr reacts with an alkenyl aryl ether: protonation of the alkene, carbocation generation, possible intramolecular attack or bromide capture, and cyclization vs addition pathways.'
    cot_agent3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    cot_inputs3 = [taskInfo, thinking2, answer2]
    thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, outlining mechanism, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], 'Critically evaluate the mechanism outline for completeness and chemical accuracy, and point out any issues.', i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == 'True':
            break
        cot_inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction3, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refining mechanism, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print('Subtask 3 answer: ', sub_tasks[-1])
    cot_instruction4 = 'Sub-task 4: Predict all plausible major products: ring-closure to chromane or dihydrobenzofuran vs simple alkyl bromide-ether.'
    cot_agent4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, predicting products, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print('Subtask 4 answer: ', sub_tasks[-1])
    cot_instruction5 = 'Sub-task 5: List and describe the structures of the provided answer choices by name and key features for all seven options.'
    cot_agent5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent5([taskInfo], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, listing choices, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print('Subtask 5 answer: ', sub_tasks[-1])
    debate_instruction6 = 'Sub-task 6: Compare predicted product skeletons to listed choices, checking ring system, methyl pattern, and bromine presence.'
    debate_agents6 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking6 = [[] for _ in range(self.max_round)]
    all_answer6 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking4, answer4, thinking5, answer5], debate_instruction6, r, is_sub_task=True)
            else:
                thinking6, answer6 = await agent([taskInfo, thinking4, answer4, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1], debate_instruction6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing structures, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision6 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision6([taskInfo] + all_thinking6[-1] + all_answer6[-1], 'Sub-task 6: Decide which choices match predicted products.', is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision6.id}, selecting matches, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print('Subtask 6 answer: ', sub_tasks[-1])
    cot_instruction7 = 'Sub-task 7: Select the choice(s) whose structures exactly match the predicted cyclized products and rule out the rest.'
    cot_agent7 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent7([taskInfo, thinking6, answer6], cot_instruction7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent7.id}, selecting final choices, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print('Subtask 7 answer: ', sub_tasks[-1])
    cot_instruction8 = 'Sub-task 8: Summarize reasoning linking mechanism, predicted products, and selected choices, justifying why other choices are not formed.'
    cot_agent8 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent8([taskInfo, thinking3, answer3, thinking4, answer4, thinking7, answer7], cot_instruction8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent8.id}, summarizing reasoning, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print('Subtask 8 answer: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer