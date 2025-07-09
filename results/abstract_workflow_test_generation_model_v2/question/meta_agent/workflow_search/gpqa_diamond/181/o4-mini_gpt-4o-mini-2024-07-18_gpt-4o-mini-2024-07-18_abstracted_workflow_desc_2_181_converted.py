async def forward_181(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_agent1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    instruction1 = "Sub-task 1: Identify and list the fundamental physical assumptions underlying the Mott–Gurney equation (space-charge-limited current), including trap-free transport, single-carrier operation, contact injection behavior, and diffusion current significance."
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": instruction1, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent1([taskInfo], instruction1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, identify fundamental assumptions, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    N = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    instruction2 = "Sub-task 2: Define and clarify each assumption in precise semiconductor-device terms: what 'trap-free' entails, criteria for 'single-carrier' operation, meaning of 'Ohmic' versus 'Schottky' contacts (injection barrier), and when diffusion current is negligible."
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": instruction2, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents2[i]([taskInfo, thinking1, answer1], instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, defining assumptions, thinking: {thinking2.content}; answer: {answer2.content}")
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
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    instruction3 = "Sub-task 3: For each of the four provided choices (A–D), map its stated conditions onto the set of assumptions defined in Sub-task 2, noting any matches or violations."
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": instruction3, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "CoT"}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, map choices onto assumptions, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    instruction4 = "Sub-task 4: Compare the mapping results to determine which choice fully satisfies all Mott–Gurney validity conditions without contradiction and select the correct statement."
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": instruction4, "context": ["user query", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "Debate"}
    for r in range(N_max):
        for agent in debate_agents:
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], instruction4, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking3, answer3] + all_thinking[r-1] + all_answer[r-1]
                thinking4, answer4 = await agent(inputs, instruction4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing mapping to select choice, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking[r].append(thinking4)
            all_answer[r].append(answer4)
    final_decision = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision([taskInfo] + all_thinking[-1] + all_answer[-1], "Sub-task 4: Make final decision on which choice is valid.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct choice, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs