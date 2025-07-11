async def forward_169(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = "Subtask 1: Define the given spin state vector ψ = (3i, 4)ᵀ from the problem statement."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_instruction,"context":["user query"],"agent_collaboration":"CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, define spin state vector, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    subtask_desc1['response']={"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)

    cot_sc_instruction = "Subtask 2: Compute the norm of ψ and normalize it to obtain a unit spin state vector ψ_norm."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, normalize spin state, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content]=thinking2_i
        answermapping[answer2_i.content]=answer2_i
    answer2_content=Counter(possible_answers).most_common(1)[0][0]
    thinking2, answer2=thinkingmapping[answer2_content], answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    subtask_desc2['response']={"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)

    cot_instruction = "Subtask 3: Write down the Pauli matrix σ_y in matrix form as [[0, -i], [i, 0]]."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_instruction,"context":["user query"],"agent_collaboration":"CoT"}
    thinking3, answer3 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, write Pauli matrix, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    subtask_desc3['response']={"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)

    cot_sc_instruction = "Subtask 4: Construct the spin operator S_y = (ħ/2)*σ_y using the matrix from subtask 3."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_sc_instruction,"context":["user query","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking4_i, answer4_i = await cot_agents[i]([taskInfo, thinking3, answer3], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, construct S_y operator, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_answers.append(answer4_i.content)
        thinkingmapping[answer4_i.content]=thinking4_i
        answermapping[answer4_i.content]=answer4_i
    answer4_content=Counter(possible_answers).most_common(1)[0][0]
    thinking4, answer4=thinkingmapping[answer4_content], answermapping[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    subtask_desc4['response']={"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)

    cot_instruction = "Subtask 5: Form the bra vector ⟨ψ_norm| by taking the conjugate transpose of the normalized state from subtask 2."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":cot_instruction,"context":["user query","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"CoT"}
    thinking5, answer5 = await cot_agent([taskInfo, thinking2, answer2], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, form bra vector, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    subtask_desc5['response']={"thinking":thinking5,"answer":answer5}
    logs.append(subtask_desc5)

    cot_sc_instruction = "Subtask 6: Compute the intermediate vector v = S_y · ψ_norm using the result of subtask 4 and ψ_norm."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc6 = {"subtask_id":"subtask_6","instruction":cot_sc_instruction,"context":["user query","thinking of subtask 4","answer of subtask 4","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking6_i, answer6_i = await cot_agents[i]([taskInfo, thinking4, answer4, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, apply operator S_y, thinking: {thinking6_i.content}; answer: {answer6_i.content}")
        possible_answers.append(answer6_i.content)
        thinkingmapping[answer6_i.content]=thinking6_i
        answermapping[answer6_i.content]=answer6_i
    answer6_content=Counter(possible_answers).most_common(1)[0][0]
    thinking6, answer6=thinkingmapping[answer6_content], answermapping[answer6_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    subtask_desc6['response']={"thinking":thinking6,"answer":answer6}
    logs.append(subtask_desc6)

    cot_instruction = "Subtask 7: Compute the expectation value ⟨ψ_norm|S_y|ψ_norm⟩ by taking the inner product ⟨ψ_norm| · v."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {"subtask_id":"subtask_7","instruction":cot_instruction,"context":["user query","thinking of subtask 5","answer of subtask 5","thinking of subtask 6","answer of subtask 6"],"agent_collaboration":"CoT"}
    thinking7, answer7 = await cot_agent([taskInfo, thinking5, answer5, thinking6, answer6], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, compute expectation value, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])
    subtask_desc7['response']={"thinking":thinking7,"answer":answer7}
    logs.append(subtask_desc7)

    debate_instruction = "Subtask 8: Simplify the algebraic result for ⟨S_y⟩, compare it to the provided choices, and select the matching letter (A, B, C, or D)."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    subtask_desc8 = {"subtask_id":"subtask_8","instruction":debate_instruction,"context":["user query","thinking of subtask 7","answer of subtask 7"],"agent_collaboration":"Debate"}
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r==0:
                thinking8_i, answer8_i = await agent([taskInfo, thinking7, answer7], debate_instruction, r, is_sub_task=True)
            else:
                thinking8_i, answer8_i = await agent([taskInfo, thinking7, answer7]+all_thinking[r-1]+all_answer[r-1], debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, simplify and match choice, thinking: {thinking8_i.content}; answer: {answer8_i.content}")
            all_thinking[r].append(thinking8_i)
            all_answer[r].append(answer8_i)
    final_decision_agent = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent([taskInfo]+all_thinking[-1]+all_answer[-1], "Subtask 8: Make final decision on the matching choice.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding matching choice, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])
    subtask_desc8['response']={"thinking":thinking8,"answer":answer8}
    logs.append(subtask_desc8)

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs