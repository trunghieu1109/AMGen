async def forward_24(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = "Sub-task 1: Introduce and define the new variables a = log₂(x), b = log₂(y), c = log₂(z) to convert the given logarithmic system into a linear system. Reflexion: Notation is consistent; moving to next subtask."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, introducing variables a, b, c, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 2: Rewrite each equation in terms of a, b, and c to obtain the linear system: a - b - c = 1/2; -a + b - c = 1/3; -a - b + c = 1/4. Reflexion: Notation is consistent; moving to next subtask."
    N = self.max_sc
    sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2, answer2 = await sc_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents[i].id}, rewriting equations, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinking_mapping[answer2.content] = thinking2
        answer_mapping[answer2.content] = answer2
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_mapping[answer2_content]
    answer2 = answer_mapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction3a = "Sub-task 3a: Set up the augmented matrix for the linear system: [1 -1 -1 | 1/2]; [-1 1 -1 | 1/3]; [-1 -1 1 | 1/4]. Reflexion: Notation is consistent; moving to next subtask."
    cot_agent3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3a = {"subtask_id": "subtask_3a", "instruction": cot_instruction3a, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "CoT"}
    thinking3a, answer3a = await cot_agent3a([taskInfo, thinking2, answer2], cot_instruction3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3a.id}, setting up augmented matrix, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a["response"] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])

    debate_instruction3b = "Sub-task 3b: Solve for variable a using row operations or elimination on the augmented matrix. Debate: compare two elimination methods and choose the correct result. Reflexion: Notation is consistent; moving to next subtask."
    debate_agents3b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking3b = [[] for _ in range(N_max)]
    all_answer3b = [[] for _ in range(N_max)]
    subtask_desc3b = {"subtask_id": "subtask_3b", "instruction": debate_instruction3b, "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"], "agent_collaboration": "Debate"}
    for r in range(N_max):
        for i, agent in enumerate(debate_agents3b):
            if r == 0:
                thinking3b_i, answer3b_i = await agent([taskInfo, thinking3a, answer3a], debate_instruction3b, r, is_sub_task=True)
            else:
                input_infos3b = [taskInfo, thinking3a, answer3a] + all_thinking3b[r-1] + all_answer3b[r-1]
                thinking3b_i, answer3b_i = await agent(input_infos3b, debate_instruction3b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, solving for a, thinking: {thinking3b_i.content}; answer: {answer3b_i.content}")
            all_thinking3b[r].append(thinking3b_i)
            all_answer3b[r].append(answer3b_i)
    final_decision3b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3b, answer3b = await final_decision3b([taskInfo] + all_thinking3b[-1] + all_answer3b[-1], "Sub-task 3b: Make final decision on the solution for a.", is_sub_task=True)
    agents.append(f"Final Decision agent, solution for a, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b["response"] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])

    cot_instruction3c = "Sub-task 3c: Solve for variable b using the chosen elimination method and the intermediate result for a. Reflexion: Notation is consistent; moving to next subtask."
    cot_agent3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3c = {"subtask_id": "subtask_3c", "instruction": cot_instruction3c, "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"], "agent_collaboration": "CoT"}
    thinking3c, answer3c = await cot_agent3c([taskInfo, thinking3b, answer3b], cot_instruction3c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3c.id}, solving for b, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c["response"] = {"thinking": thinking3c, "answer": answer3c}
    logs.append(subtask_desc3c)
    print("Step 3c: ", sub_tasks[-1])

    cot_instruction3d = "Sub-task 3d: Solve for variable c using the previous values of a and b. Reflexion: Notation is consistent; moving to next subtask."
    cot_agent3d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3d = {"subtask_id": "subtask_3d", "instruction": cot_instruction3d, "context": ["user query", "thinking of subtask 3c", "answer of subtask 3c"], "agent_collaboration": "CoT"}
    thinking3d, answer3d = await cot_agent3d([taskInfo, thinking3b, answer3b, thinking3c, answer3c], cot_instruction3d, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3d.id}, solving for c, thinking: {thinking3d.content}; answer: {answer3d.content}")
    sub_tasks.append(f"Sub-task 3d output: thinking - {thinking3d.content}; answer - {answer3d.content}")
    subtask_desc3d["response"] = {"thinking": thinking3d, "answer": answer3d}
    logs.append(subtask_desc3d)
    print("Step 3d: ", sub_tasks[-1])

    cot_instruction4 = "Sub-task 4: Express the target log₂(x^4 y^3 z^2) in terms of a, b, c as 4a + 3b + 2c. Reflexion: Notation is consistent; moving to next subtask."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_instruction4, "context": ["user query", "thinking of subtask 3d", "answer of subtask 3d"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3d, answer3d], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, expressing target, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction5 = "Sub-task 5: Substitute the computed values of a, b, c into the expression 4a + 3b + 2c and simplify to obtain a single fractional value. Reflexion: Notation is consistent; moving to next subtask."
    sc_agents5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers5 = []
    thinking_map5 = {}
    answer_map5 = {}
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_sc_instruction5, "context": ["user query", "thinking of subtask 4", "answer of subtask 4"], "agent_collaboration": "SC_CoT"}
    for i in range(self.max_sc):
        thinking5, answer5 = await sc_agents5[i]([taskInfo, thinking4, answer4], cot_sc_instruction5, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents5[i].id}, simplifying expression, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers5.append(answer5.content)
        thinking_map5[answer5.content] = thinking5
        answer_map5[answer5.content] = answer5
    answer5_content = Counter(possible_answers5).most_common(1)[0][0]
    thinking5 = thinking_map5[answer5_content]
    answer5 = answer_map5[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction6 = "Sub-task 6: Reduce the numerical result of 4a + 3b + 2c to lowest terms m/n, then compute and output m + n. Reflexion: Notation is consistent; workflow complete."
    cot_agent6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot_instruction6, "context": ["user query", "thinking of subtask 5", "answer of subtask 5"], "agent_collaboration": "CoT"}
    thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5], cot_instruction6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent6.id}, reducing fraction and computing m+n, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6["response"] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs