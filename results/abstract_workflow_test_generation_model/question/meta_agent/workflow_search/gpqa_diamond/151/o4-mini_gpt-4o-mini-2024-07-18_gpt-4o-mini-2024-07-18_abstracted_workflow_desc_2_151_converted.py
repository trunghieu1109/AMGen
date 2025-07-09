async def forward_151(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Sub-task 1: Identify the source of the peptide, the eukaryotic model organism, and the observed phenotype from the query."
    cot_agent1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction1, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, identifying context, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers2 = []
    thinking_map2 = {}
    answer_map2 = {}
    cot_instruction2 = "Sub-task 2: From the context, determine the experimental assay and its target for profiling proteins in the shmoo active chromatin."
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_instruction2, "context": ["user query", thinking1.content, answer1.content], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents2:
        t2, a2 = await agent([taskInfo, thinking1, answer1], cot_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, determining assay, thinking: {t2.content}; answer: {a2.content}")
        possible_answers2.append(a2.content)
        thinking_map2[a2.content] = t2
        answer_map2[a2.content] = a2
    ans2_content = Counter(possible_answers2).most_common(1)[0][0]
    thinking2 = thinking_map2[ans2_content]
    answer2 = answer_map2[ans2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction3 = "Sub-task 3: List the candidate protein complexes from the multiple-choice options."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_instruction3, "context": ["user query", thinking2.content, answer2.content], "agent_collaboration": "CoT"}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, listing complexes, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    N4 = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_answers4 = []
    thinking_map4 = {}
    answer_map4 = {}
    cot_instruction4 = "Sub-task 4: Assess which listed complexes associate with actively transcribed chromatin in yeast shmoo."
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_instruction4, "context": ["user query", thinking3.content, answer3.content], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents4:
        t4, a4 = await agent([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, assessing complexes, thinking: {t4.content}; answer: {a4.content}")
        possible_answers4.append(a4.content)
        thinking_map4[a4.content] = t4
        answer_map4[a4.content] = a4
    ans4_content = Counter(possible_answers4).most_common(1)[0][0]
    thinking4 = thinking_map4[ans4_content]
    answer4 = answer_map4[ans4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_instruction5 = "Sub-task 5: Determine the cell-cycle state of shmoo-forming yeast and its effect on replication complex loading."
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_instruction5, "context": ["user query", thinking1.content, answer1.content], "agent_collaboration": "CoT"}
    thinking5, answer5 = await cot_agent5([taskInfo, thinking1, answer1], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, inferring G1 arrest, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_instruction6 = "Sub-task 6: Integrate active chromatin association and G1 arrest insights to evaluate relative abundance of each candidate complex."
    cot_agent6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs6 = [taskInfo, thinking4, answer4, thinking5, answer5]
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot_instruction6, "context": ["user query", thinking4.content, answer4.content, thinking5.content, answer5.content], "agent_collaboration": "Reflexion"}
    thinking6, answer6 = await cot_agent6(inputs6, cot_instruction6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent6.id}, integrating insights, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(self.max_round):
        feedback6, correct6 = await critic_agent6([taskInfo, thinking6, answer6], "Review the integration and list any limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent6.id}, feedback: {feedback6.content}; correct: {correct6.content}")
        if correct6.content == "True":
            break
        inputs6.extend([thinking6, answer6, feedback6])
        thinking6, answer6 = await cot_agent6(inputs6, cot_instruction6, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent6.id}, refined integration, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    debate_instruction7 = "Sub-task 7: Compare predicted abundances and select the complex expected to yield the fewest proteins in the active-chromatin ChIP-MS assay."
    debate_agents7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking7 = [[] for _ in range(self.max_round)]
    all_answer7 = [[] for _ in range(self.max_round)]
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": debate_instruction7, "context": ["user query", thinking6.content, answer6.content], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for agent in debate_agents7:
            if r == 0:
                t7, a7 = await agent([taskInfo, thinking6, answer6], debate_instruction7, r, is_sub_task=True)
            else:
                prev = all_thinking7[r-1] + all_answer7[r-1]
                t7, a7 = await agent([taskInfo, thinking6, answer6] + prev, debate_instruction7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {t7.content}; answer: {a7.content}")
            all_thinking7[r].append(t7)
            all_answer7[r].append(a7)
    final_decision_agent7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on which complex yields the fewest proteins.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent7.id}, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    cot_instruction8 = "Sub-task 8: Map the selected protein complex to its multiple-choice letter (A-D)."
    cot_agent8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {"subtask_id": "subtask_8", "instruction": cot_instruction8, "context": ["user query", thinking7.content, answer7.content], "agent_collaboration": "CoT"}
    thinking8, answer8 = await cot_agent8([taskInfo, thinking7, answer7], cot_instruction8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent8.id}, mapping to letter, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs