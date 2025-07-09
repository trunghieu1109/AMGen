async def forward_166(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Compute normalization constant N = sqrt(1 + sin(2φ) * exp(-2α^2)) for φ = -π/4 and α = 0.5"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_instruction, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing N, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])
    cot_sc_instruction = "Sub-task 2: Form the cat state |ψ> = (cos φ |α> + sin φ |−α>) / N using φ, α, and N from subtask_1"
    N_sc = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers = []
    thinking_map = {}
    answer_map = {}
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': cot_sc_instruction, 'context': ['user query', 'answer of subtask_1'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N_sc):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, forming state, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers.append(answer2_i.content)
        thinking_map[answer2_i.content] = thinking2_i
        answer_map[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_map[answer2_content]
    answer2 = answer_map[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])
    ref_instruction = "Sub-task 3: Construct density matrix ρ = |ψ>⟨ψ| from the cat state"
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': ref_instruction, 'context': ['user query', 'answer of subtask_2'], 'agent_collaboration': 'Reflexion'}
    thinking3, answer3 = await cot_agent3(inputs3, ref_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, constructing ρ, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], "Review the density matrix construction and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == 'True':
            break
        inputs3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent3(inputs3, ref_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refining ρ, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])
    cot_sc_instruction4 = "Sub-task 4: Compute the first and second moments of ρ to get mean vector and covariance matrix"
    N_sc4 = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc4)]
    possible4 = []
    think_map4 = {}
    ans_map4 = {}
    subtask_desc4 = {'subtask_id': 'subtask_4', 'instruction': cot_sc_instruction4, 'context': ['user query', 'answer of subtask_3'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N_sc4):
        thinking4_i, answer4_i = await cot_agents4[i]([taskInfo, thinking3, answer3], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, computing moments, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible4.append(answer4_i.content)
        think_map4[answer4_i.content] = thinking4_i
        ans_map4[answer4_i.content] = answer4_i
    ans4_content = Counter(possible4).most_common(1)[0][0]
    thinking4 = think_map4[ans4_content]
    answer4 = ans_map4[ans4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])
    ref_instruction5 = "Sub-task 5: Build reference Gaussian state τ matching the computed moments"
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs5 = [taskInfo, thinking4, answer4]
    subtask_desc5 = {'subtask_id': 'subtask_5', 'instruction': ref_instruction5, 'context': ['user query', 'answer of subtask_4'], 'agent_collaboration': 'Reflexion'}
    thinking5, answer5 = await cot_agent5(inputs5, ref_instruction5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, constructing τ, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(self.max_round):
        feedback5, correct5 = await critic_agent5([taskInfo, thinking5, answer5], "Review τ construction and give feedback.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent5.id}, feedback: {feedback5.content}; correct: {correct5.content}")
        if correct5.content == 'True':
            break
        inputs5.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = await cot_agent5(inputs5, ref_instruction5, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, refining τ, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {'thinking': thinking5, 'answer': answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])
    cot_sc_instruction6 = "Sub-task 6: Evaluate S(rho) = -Tr[rho ln rho] (von Neumann entropy of rho)"
    N_sc6 = self.max_sc
    cot_agents6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc6)]
    possible6 = []
    think_map6 = {}
    ans_map6 = {}
    subtask_desc6 = {'subtask_id': 'subtask_6', 'instruction': cot_sc_instruction6, 'context': ['user query', 'answer of subtask_3'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N_sc6):
        thinking6_i, answer6_i = await cot_agents6[i]([taskInfo, thinking3, answer3], cot_sc_instruction6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents6[i].id}, computing S(rho), thinking: {thinking6_i.content}; answer: {answer6_i.content}")
        possible6.append(answer6_i.content)
        think_map6[answer6_i.content] = thinking6_i
        ans_map6[answer6_i.content] = answer6_i
    ans6_content = Counter(possible6).most_common(1)[0][0]
    thinking6 = think_map6[ans6_content]
    answer6 = ans_map6[ans6_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {'thinking': thinking6, 'answer': answer6}
    logs.append(subtask_desc6)
    print('Step 6: ', sub_tasks[-1])
    cot_sc_instruction7 = "Sub-task 7: Evaluate S(tau) = -Tr[tau ln tau] (von Neumann entropy of tau)"
    N_sc7 = self.max_sc
    cot_agents7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc7)]
    possible7 = []
    think_map7 = {}
    ans_map7 = {}
    subtask_desc7 = {'subtask_id': 'subtask_7', 'instruction': cot_sc_instruction7, 'context': ['user query', 'answer of subtask_5'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N_sc7):
        thinking7_i, answer7_i = await cot_agents7[i]([taskInfo, thinking5, answer5], cot_sc_instruction7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents7[i].id}, computing S(tau), thinking: {thinking7_i.content}; answer: {answer7_i.content}")
        possible7.append(answer7_i.content)
        think_map7[answer7_i.content] = thinking7_i
        ans_map7[answer7_i.content] = answer7_i
    ans7_content = Counter(possible7).most_common(1)[0][0]
    thinking7 = think_map7[ans7_content]
    answer7 = ans_map7[ans7_content]
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {'thinking': thinking7, 'answer': answer7}
    logs.append(subtask_desc7)
    print('Step 7: ', sub_tasks[-1])
    debate_instruction8 = "Sub-task 8: Compute non-Gaussianity Δ = Tr[rho ln rho] - Tr[tau ln tau] = -S(rho) + S(tau)"
    debate_agents8 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    rounds8 = self.max_round
    all_thinking8 = [[] for _ in range(rounds8)]
    all_answer8 = [[] for _ in range(rounds8)]
    subtask_desc8 = {'subtask_id': 'subtask_8', 'instruction': debate_instruction8, 'context': ['user query', 'answer of subtask_6', 'answer of subtask_7'], 'agent_collaboration': 'Debate'}
    for r in range(rounds8):
        for i, agent in enumerate(debate_agents8):
            if r == 0:
                thinking8, answer8 = await agent([taskInfo, thinking6, answer6, thinking7, answer7], debate_instruction8, r, is_sub_task=True)
            else:
                inputs8 = [taskInfo, thinking6, answer6, thinking7, answer7] + all_thinking8[r-1] + all_answer8[r-1]
                thinking8, answer8 = await agent(inputs8, debate_instruction8, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing Δ, thinking: {thinking8.content}; answer: {answer8.content}")
            all_thinking8[r].append(thinking8)
            all_answer8[r].append(answer8)
    final_decision8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision8([taskInfo] + all_thinking8[-1] + all_answer8[-1], "Sub-task 8: Decide on Δ value.", is_sub_task=True)
    agents.append(f"Final Decision agent, Δ decision, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {'thinking': thinking8, 'answer': answer8}
    logs.append(subtask_desc8)
    print('Step 8: ', sub_tasks[-1])
    cot_instruction9 = "Sub-task 9: Plug in φ = -π/4 and α = 0.5 into Δ to obtain numerical value"
    cot_agent9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc9 = {'subtask_id': 'subtask_9', 'instruction': cot_instruction9, 'context': ['user query', 'answer of subtask_8'], 'agent_collaboration': 'CoT'}
    thinking9, answer9 = await cot_agent9([taskInfo, thinking8, answer8], cot_instruction9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent9.id}, numeric Δ, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {'thinking': thinking9, 'answer': answer9}
    logs.append(subtask_desc9)
    print('Step 9: ', sub_tasks[-1])
    cot_instruction10 = "Sub-task 10: Compare computed Δ against choices [2.48, 0, 1.38, 0.25] and return corresponding letter"
    cot_agent10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc10 = {'subtask_id': 'subtask_10', 'instruction': cot_instruction10, 'context': ['user query', 'answer of subtask_9'], 'agent_collaboration': 'CoT'}
    thinking10, answer10 = await cot_agent10([taskInfo, thinking9, answer9], cot_instruction10, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent10.id}, choose answer, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {'thinking': thinking10, 'answer': answer10}
    logs.append(subtask_desc10)
    print('Step 10: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs