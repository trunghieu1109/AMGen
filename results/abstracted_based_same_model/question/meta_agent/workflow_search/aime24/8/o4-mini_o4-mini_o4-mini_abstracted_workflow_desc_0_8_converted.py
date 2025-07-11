async def forward_8(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Define the rules of the game: a pile of n tokens, players alternately remove 1 or 4 tokens, last move wins. Introduce the concepts of P-positions (losing for the player to move) and N-positions (winning for the player to move) under normal play."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, defining game rules, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    cot_sc_instruction = "Sub-task 2: Compute the P/N classification for n from 0 up to 10 using the recurrence: a position is N if it has at least one move to a P-position; otherwise it is P."
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", thinking1.content, answer1.content], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, classifying positions for n<=10, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content] = thinking2_i
        answermapping[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    reflex_instruction = "Sub-task 3: Examine the list of P-positions obtained for n ≤ 10 to identify a pattern or periodicity, and conjecture the general criterion that P-positions are exactly those with n mod 5 in {0,2}."
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": reflex_instruction, "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content], "agent_collaboration": "Reflexion"}
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent3(cot_inputs, reflex_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, identifying pattern, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback_i, correct_i = await critic_agent([taskInfo, thinking3, answer3], "Please review the conjectured criterion and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback_i.content}; answer: {correct_i.content}")
        if correct_i.content == "True":
            break
        cot_inputs.extend([thinking3, answer3, feedback_i])
        thinking3, answer3 = await cot_agent3(cot_inputs, reflex_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refining pattern identification, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    N4 = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_answers4 = []
    thinkingmapping4 = {}
    answermapping4 = {}
    instruction4 = "Sub-task 4: Using the criterion n mod 5 ∈ {0,2}, count all positive integers n ≤ 2024 that satisfy this criterion."
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": instruction4, "context": ["user query", thinking3.content, answer3.content], "agent_collaboration": "SC_CoT"}
    for i in range(N4):
        thinking4_i, answer4_i = await cot_agents4[i]([taskInfo, thinking3, answer3], instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, counting n≤2024 with n mod5 in {0,2}, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_answers4.append(answer4_i.content)
        thinkingmapping4[answer4_i.content] = thinking4_i
        answermapping4[answer4_i.content] = answer4_i
    answer4_content = Counter(possible_answers4).most_common(1)[0][0]
    thinking4 = thinkingmapping4[answer4_content]
    answer4 = answermapping4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent([taskInfo, thinking4, answer4], "Sub-task 5: Return the final count as the answer to the query.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent.id}, providing final count, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": "Sub-task 5: Return the final count as the answer to the query.", "context": ["user query", thinking4.content, answer4.content], "agent_collaboration": "Final Decision"}
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs