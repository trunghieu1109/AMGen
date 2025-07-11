async def forward_160(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Sub-task 1: Parse the query to extract vacuum pressure (P < 10^-9 Torr), temperature (T), compartment volume (if given), and electron beam energy (1000 kV)."
    cot_agent1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_instruction1,"context":["user query"],"agent_collaboration":"CoT"}
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, parsing parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction2 = "Sub-task 2: Compute the residual gas number density n = P / (k_B · T) using the extracted pressure and temperature values."
    N = self.max_sc
    sc_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers2 = []
    thinking_map2 = {}
    answer_map2 = {}
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction2,"context":["user query","thinking1","answer1"],"agent_collaboration":"SC_CoT"}
    for agent in sc_agents2:
        thinking2_i, answer2_i = await agent([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, computing n, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers2.append(answer2_i.content)
        thinking_map2[answer2_i.content] = thinking2_i
        answer_map2[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers2).most_common(1)[0][0]
    thinking2 = thinking_map2[answer2_content]
    answer2 = answer_map2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction3 = "Sub-task 3: Lookup or select a representative molecular collision cross section σ_m for gas–gas interactions under ultra–high vacuum conditions."
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_instruction3,"context":["user query"],"agent_collaboration":"CoT"}
    thinking3, answer3 = await cot_agent3([taskInfo], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, selecting σ_m, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_sc_instruction4 = "Sub-task 4: Calculate the molecular mean free path λ1 = 1 / (n · σ_m) using the outputs of subtask 2 and subtask 3."
    sc_agents4 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers4 = []
    thinking_map4 = {}
    answer_map4 = {}
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_sc_instruction4,"context":["user query","thinking2","answer2","thinking3","answer3"],"agent_collaboration":"SC_CoT"}
    for agent in sc_agents4:
        thinking4_i, answer4_i = await agent([taskInfo, thinking2, answer2, thinking3, answer3], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, calculating λ1, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_answers4.append(answer4_i.content)
        thinking_map4[answer4_i.content] = thinking4_i
        answer_map4[answer4_i.content] = answer4_i
    answer4_content = Counter(possible_answers4).most_common(1)[0][0]
    thinking4 = thinking_map4[answer4_content]
    answer4 = answer_map4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_instruction5a = "Sub-task 5a: Research and retrieve an electron–molecule scattering cross section σ_e at 1000 kV from independent literature sources."
    cot_agent5a = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5a = {"subtask_id":"subtask_5a","instruction":cot_instruction5a,"context":["user query"],"agent_collaboration":"CoT"}
    thinking5a, answer5a = await cot_agent5a([taskInfo], cot_instruction5a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5a.id}, retrieving σ_e, thinking: {thinking5a.content}; answer: {answer5a.content}")
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {"thinking":thinking5a,"answer":answer5a}
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    cot_reflect_instruction5b = "Sub-task 5b: Document and verify the provenance of σ_e to ensure independence from λ calculations."
    cot_agent5b = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent5b = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc5b = {"subtask_id":"subtask_5b","instruction":cot_reflect_instruction5b,"context":["user query","thinking5a","answer5a"],"agent_collaboration":"Reflexion"}
    thinking5b, answer5b = await cot_agent5b([taskInfo, thinking5a, answer5a], cot_reflect_instruction5b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5b.id}, documenting σ_e provenance, thinking: {thinking5b.content}; answer: {answer5b.content}")
    for i in range(self.max_round):
        feedback5b, correct5b = await critic_agent5b([taskInfo, thinking5b, answer5b], "Review the provenance documentation for σ_e and flag any issues.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent5b.id}, feedback: {feedback5b.content}; correct: {correct5b.content}")
        if correct5b.content == "True":
            break
        thinking5b, answer5b = await cot_agent5b([taskInfo, thinking5a, answer5a, feedback5b], cot_reflect_instruction5b, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5b.id}, refined documentation, thinking: {thinking5b.content}; answer: {answer5b.content}")
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {"thinking":thinking5b,"answer":answer5b}
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    cot_sc_instruction6 = "Sub-task 6: Compute the electron scattering mean free path λ2 = 1 / (n · σ_e) using n from subtask 2 and verified σ_e from subtask 5b."
    sc_agents6 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers6 = []
    thinking_map6 = {}
    answer_map6 = {}
    subtask_desc6 = {"subtask_id":"subtask_6","instruction":cot_sc_instruction6,"context":["user query","thinking2","answer2","thinking5b","answer5b"],"agent_collaboration":"SC_CoT"}
    for agent in sc_agents6:
        thinking6_i, answer6_i = await agent([taskInfo, thinking2, answer2, thinking5b, answer5b], cot_sc_instruction6, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, computing λ2, thinking: {thinking6_i.content}; answer: {answer6_i.content}")
        possible_answers6.append(answer6_i.content)
        thinking_map6[answer6_i.content] = thinking6_i
        answer_map6[answer6_i.content] = answer6_i
    answer6_content = Counter(possible_answers6).most_common(1)[0][0]
    thinking6 = thinking_map6[answer6_content]
    answer6 = answer_map6[answer6_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking":thinking6,"answer":answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    debate_instruction7 = "Sub-task 7: Debate the expected ordering of λ1 and λ2 based on the principle that σ_e < σ_m, refining physical interpretation."
    debate_agents7 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking7 = [[] for _ in range(self.max_round)]
    all_answer7 = [[] for _ in range(self.max_round)]
    subtask_desc7 = {"subtask_id":"subtask_7","instruction":debate_instruction7,"context":["user query","thinking4","answer4","thinking6","answer6"],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for agent in debate_agents7:
            if r == 0:
                thinking7_i, answer7_i = await agent([taskInfo, thinking4, answer4, thinking6, answer6], debate_instruction7, r, is_sub_task=True)
            else:
                inputs7 = [taskInfo, thinking4, answer4, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7_i, answer7_i = await agent(inputs7, debate_instruction7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking7_i.content}; answer: {answer7_i.content}")
            all_thinking7[r].append(thinking7_i)
            all_answer7[r].append(answer7_i)
    final_agent7 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_agent7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final conclusion on λ ordering.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking":thinking7,"answer":answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    cot_reflect_instruction8 = "Sub-task 8: Perform a physical plausibility check by comparing R = λ2 / λ1 against expectation R > 1."
    cot_agent8 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent8 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {"subtask_id":"subtask_8","instruction":cot_reflect_instruction8,"context":["user query","thinking4","answer4","thinking6","answer6"],"agent_collaboration":"Reflexion"}
    thinking8, answer8 = await cot_agent8([taskInfo, thinking4, answer4, thinking6, answer6], cot_reflect_instruction8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent8.id}, checking plausibility, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(self.max_round):
        feedback8, correct8 = await critic_agent8([taskInfo, thinking8, answer8], "Review the plausibility check of R and flag inconsistencies.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent8.id}, feedback: {feedback8.content}; correct: {correct8.content}")
        if correct8.content == "True":
            break
        thinking8, answer8 = await cot_agent8([taskInfo, thinking4, answer4, thinking6, answer6, feedback8], cot_reflect_instruction8, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent8.id}, refined check, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking":thinking8,"answer":answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    cot_sc_instruction9 = "Sub-task 9: Determine which inequality holds—(λ2 = λ1), (λ2 < λ1), (λ1 < λ2 < 1.22·λ1), or (λ2 ≥ 1.22·λ1)—based on validated λ1 and λ2."
    sc_agents9 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers9 = []
    thinking_map9 = {}
    answer_map9 = {}
    subtask_desc9 = {"subtask_id":"subtask_9","instruction":cot_sc_instruction9,"context":["user query","thinking4","answer4","thinking6","answer6"],"agent_collaboration":"SC_CoT"}
    for agent in sc_agents9:
        thinking9_i, answer9_i = await agent([taskInfo, thinking4, answer4, thinking6, answer6], cot_sc_instruction9, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, selecting inequality, thinking: {thinking9_i.content}; answer: {answer9_i.content}")
        possible_answers9.append(answer9_i.content)
        thinking_map9[answer9_i.content] = thinking9_i
        answer_map9[answer9_i.content] = answer9_i
    answer9_content = Counter(possible_answers9).most_common(1)[0][0]
    thinking9 = thinking_map9[answer9_content]
    answer9 = answer_map9[answer9_content]
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking":thinking9,"answer":answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])
    debate_instruction10 = "Sub-task 10: Map the identified inequality relationship to its corresponding multiple-choice letter (A, B, C, or D)."
    debate_agents10 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking10 = [[] for _ in range(self.max_round)]
    all_answer10 = [[] for _ in range(self.max_round)]
    subtask_desc10 = {"subtask_id":"subtask_10","instruction":debate_instruction10,"context":["user query","thinking9","answer9"],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for agent in debate_agents10:
            if r == 0:
                thinking10_i, answer10_i = await agent([taskInfo, thinking9, answer9], debate_instruction10, r, is_sub_task=True)
            else:
                inputs10 = [taskInfo, thinking9, answer9] + all_thinking10[r-1] + all_answer10[r-1]
                thinking10_i, answer10_i = await agent(inputs10, debate_instruction10, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking10_i.content}; answer: {answer10_i.content}")
            all_thinking10[r].append(thinking10_i)
            all_answer10[r].append(answer10_i)
    final_agent10 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking10, answer10 = await final_agent10([taskInfo] + all_thinking10[-1] + all_answer10[-1], "Sub-task 10: Make final mapping decision.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking":thinking10,"answer":answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs