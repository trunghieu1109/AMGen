async def forward_11(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    cot_instruction_0_1 = "Sub-task 0_1: Define the structure of lattice paths from (0,0) to (8,8) with exactly four direction changes, emphasizing that such paths consist of exactly five alternating monotone segments of right and up steps."
    subtask_desc_0_1 = {
        "subtask_id": "subtask_0_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, defining lattice path structure, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0_1: ", sub_tasks[-1])

    cot_instruction_0_2 = "Sub-task 0_2: Clarify the meaning of a direction change as a switch from right to up or up to right step, and confirm that the path length and step counts imply 8 right and 8 up steps in total."
    subtask_desc_0_2 = {
        "subtask_id": "subtask_0_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, clarifying direction change meaning, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0_2: ", sub_tasks[-1])

    debate_instruction_0_3 = "Sub-task 0_3: Determine the possible starting directions (right or up) for the path and establish that both cases must be considered separately to cover all valid paths with exactly four direction changes. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    all_thinking_0_3 = []
    all_answer_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "subtask_0_3",
        "instruction": debate_instruction_0_3,
        "context": ["user query", thinking_0_1, thinking_0_2],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_0_3):
        thinking_0_3, answer_0_3 = await agent([taskInfo, thinking_0_1, thinking_0_2], debate_instruction_0_3, i, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, determining starting direction, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
        all_thinking_0_3.append(thinking_0_3)
        all_answer_0_3.append(answer_0_3)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_0_3 = "Sub-task 0_3: Your problem is to determine the possible starting directions for the path with exactly four direction changes. Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + all_thinking_0_3, final_instr_0_3, is_sub_task=True)
    agents.append(f"Final Decision agent, determining starting directions, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0_3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0_3: ", sub_tasks[-1])

    cot_sc_instruction_1_1 = "Sub-task 1_1: Formulate the problem of counting valid paths as counting integer compositions of 8 right steps and 8 up steps into five positive parts alternating between R and U, depending on the starting direction."
    N_sc = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_3],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_1, answer_1_1 = await cot_agents_1_1[i]([taskInfo, thinking_0_3], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, formulating counting problem, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1_1 = "Sub-task 1_1: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_thinkings_1_1, final_instr_1_1, is_sub_task=True)
    agents.append(f"Final Decision agent, formulating counting problem, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1_1: ", sub_tasks[-1])

    cot_instruction_1_2 = "Sub-task 1_2: Enumerate all possible positive integer partitions of 8 into either 3 or 2 parts (depending on whether the path starts with R or U) for the runs of R and U steps respectively, ensuring the total sums match 8 for each direction."
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "subtask_1_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, enumerating integer partitions, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1_2: ", sub_tasks[-1])

    cot_sc_instruction_1_3 = "Sub-task 1_3: Calculate the number of ways to assign lengths to the five runs (alternating R and U) that sum to 8 for each direction, using combinatorial formulas for compositions of integers into positive parts."
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "subtask_1_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_1_2],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_3, answer_1_3 = await cot_agents_1_3[i]([taskInfo, thinking_1_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, calculating run-length assignments, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
        possible_answers_1_3.append(answer_1_3)
        possible_thinkings_1_3.append(thinking_1_3)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1_3 = "Sub-task 1_3: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_thinkings_1_3, final_instr_1_3, is_sub_task=True)
    agents.append(f"Final Decision agent, calculating run-length assignments, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1_3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1_3: ", sub_tasks[-1])

    debate_instruction_2_1 = "Sub-task 2_1: Combine the counts of valid run-length assignments for both starting directions to find the total number of lattice paths with exactly four direction changes from (0,0) to (8,8). Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    all_thinking_2_1 = []
    all_answer_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_1_3],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_2_1):
        thinking_2_1, answer_2_1 = await agent([taskInfo, thinking_1_3], debate_instruction_2_1, i, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, combining counts, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
        all_thinking_2_1.append(thinking_2_1)
        all_answer_2_1.append(answer_2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2_1 = "Sub-task 2_1: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1, final_instr_2_1, is_sub_task=True)
    agents.append(f"Final Decision agent, combining counts, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2_1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    return final_answer, logs
