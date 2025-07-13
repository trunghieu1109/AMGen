async def forward_11(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: SC_CoT for run-pattern identification
    cot_sc_instruction = (
        "Sub-task 1: Model each path as a binary word of 8 Es and 8 Ns and show that exactly four direction changes imply exactly five runs; "
        "identify the two possible run patterns (starting with E or starting with N)."
    )
    cot_agents = [
        LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(self.max_sc)
    ]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents:
        thinking, answer = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_decision_agent_1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1(
        [taskInfo] + possible_thinkings + possible_answers,
        "Sub-task 1: Synthesize most consistent answer. Given all above thinking and answers, find the most consistent run patterns with exactly four turns.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2, Subtask 2: Debate for pattern starting with E
    debate_instr = (
        "Given solutions from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_instruction_2 = (
        "Sub-task 2: For the pattern starting with E (runs E–N–E–N–E), count positive integer solutions to r1+r3+r5=8 and r2+r4=8. "
        + debate_instr
    )
    debate_agents_2 = [
        LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_2 = self.max_round
    all_thinkings2 = [[] for _ in range(N_max_2)]
    all_answers2 = [[] for _ in range(N_max_2)]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instruction_2,
        "context": ["user query","thinking of subtask 1","answer of subtask 1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking2, answer2 = await agent([
                    taskInfo, thinking1, answer1
                ], debate_instruction_2, r, is_sub_task=True)
            else:
                thinking2, answer2 = await agent(
                    [taskInfo, thinking1, answer1] + all_thinkings2[r-1] + all_answers2[r-1],
                    debate_instruction_2,
                    r,
                    is_sub_task=True
                )
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinkings2[r].append(thinking2)
            all_answers2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2(
        [taskInfo, thinking1, answer1] + all_thinkings2[-1] + all_answers2[-1],
        "Sub-task 2: Given all above thinking and answers, provide a final count for the pattern starting with E.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 2, Subtask 3: Debate for pattern starting with N
    debate_instruction_3 = (
        "Sub-task 3: For the pattern starting with N (runs N–E–N–E–N), count positive integer solutions to r1+r3+r5=8 and r2+r4=8. "
        + debate_instr
    )
    debate_agents_3 = [
        LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_3 = self.max_round
    all_thinkings3 = [[] for _ in range(N_max_3)]
    all_answers3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query","thinking of subtask 1","answer of subtask 1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([
                    taskInfo, thinking1, answer1
                ], debate_instruction_3, r, is_sub_task=True)
            else:
                thinking3, answer3 = await agent(
                    [taskInfo, thinking1, answer1] + all_thinkings3[r-1] + all_answers3[r-1],
                    debate_instruction_3,
                    r,
                    is_sub_task=True
                )
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinkings3[r].append(thinking3)
            all_answers3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3(
        [taskInfo, thinking1, answer1] + all_thinkings3[-1] + all_answers3[-1],
        "Sub-task 3: Given all above thinking and answers, provide a final count for the pattern starting with N.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 3: CoT for summation of results
    cot_instruction4 = (
        "Sub-task 4: Sum the counts from the two starting-direction cases to obtain the total number of length-16 paths with exactly four turns."
    )
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction4,
        "context": ["user query","answer of subtask 2","answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent4(
        [taskInfo, answer2, answer3], cot_instruction4, is_sub_task=True
    )
    agents.append(f"CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs