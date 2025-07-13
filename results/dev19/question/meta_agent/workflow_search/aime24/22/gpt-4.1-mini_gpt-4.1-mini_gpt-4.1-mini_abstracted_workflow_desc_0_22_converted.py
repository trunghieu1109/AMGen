async def forward_22(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = "Sub-task 1: Identify and clearly state the domain of the problem: all lists of positive integers whose sum is 30, with context from the user query."
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, analyzing domain, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    debate_instruction_0_2 = "Sub-task 2: Analyze and characterize the implications of the unique mode being 9, including frequency constraints on 9 and other numbers, with context from Sub-task 0.1 output and user query. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    all_thinking_0_2 = []
    all_answer_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": debate_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_0_2):
        thinking_i, answer_i = await agent([taskInfo, thinking_0_1], debate_instruction_0_2, i, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, analyzing mode implications, thinking: {thinking_i.content}; answer: {answer_i.content}")
        all_thinking_0_2.append(thinking_i)
        all_answer_0_2.append(answer_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + all_thinking_0_2, "Sub-task 2: Synthesize and choose the most consistent answer for mode implications. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing mode implications, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    debate_instruction_0_3 = "Sub-task 3: Determine the implications of the median being a positive integer not in the list, including deducing that the list length must be even and the median is the average of the two middle elements, with context from Sub-task 0.1 output and user query. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    all_thinking_0_3 = []
    all_answer_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": debate_instruction_0_3,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_0_3):
        thinking_i, answer_i = await agent([taskInfo, thinking_0_1], debate_instruction_0_3, i, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, analyzing median implications, thinking: {thinking_i.content}; answer: {answer_i.content}")
        all_thinking_0_3.append(thinking_i)
        all_answer_0_3.append(answer_i)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + all_thinking_0_3, "Sub-task 3: Synthesize and choose the most consistent answer for median implications. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing median implications, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    cot_instruction_0_4 = "Sub-task 4: Establish possible list lengths consistent with the median condition and sum constraint, considering even lengths and positive integers, with context from Sub-task 0.3 and 0.1 outputs."
    cot_agent_0_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_4 = {
        "subtask_id": "stage_0.subtask_4",
        "instruction": cot_instruction_0_4,
        "context": ["user query", thinking_0_3.content, thinking_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    thinking_0_4, answer_0_4 = await cot_agent_0_4([taskInfo, thinking_0_3, thinking_0_1], cot_instruction_0_4, is_sub_task=True)
    agents.append(f"CoT-SC agent {cot_agent_0_4.id}, establishing possible list lengths, thinking: {thinking_0_4.content}; answer: {answer_0_4.content}")
    sub_tasks.append(f"Sub-task 0.4 output: thinking - {thinking_0_4.content}; answer - {answer_0_4.content}")
    subtask_desc_0_4['response'] = {"thinking": thinking_0_4, "answer": answer_0_4}
    logs.append(subtask_desc_0_4)
    print("Step 0.4: ", sub_tasks[-1])

    cot_instruction_0_5 = "Sub-task 5: Formulate constraints on the list elements based on the sum, mode, and median conditions, preparing for enumeration of valid lists, with context from Sub-task 0.2, 0.3, and 0.4 outputs."
    cot_agent_0_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_5 = {
        "subtask_id": "stage_0.subtask_5",
        "instruction": cot_instruction_0_5,
        "context": ["user query", thinking_0_2.content, thinking_0_3.content, thinking_0_4.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_5, answer_0_5 = await cot_agent_0_5([taskInfo, thinking_0_2, thinking_0_3, thinking_0_4], cot_instruction_0_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_5.id}, formulating constraints, thinking: {thinking_0_5.content}; answer: {answer_0_5.content}")
    sub_tasks.append(f"Sub-task 0.5 output: thinking - {thinking_0_5.content}; answer - {answer_0_5.content}")
    subtask_desc_0_5['response'] = {"thinking": thinking_0_5, "answer": answer_0_5}
    logs.append(subtask_desc_0_5)
    print("Step 0.5: ", sub_tasks[-1])

    debate_instruction_1_1 = "Sub-task 1: Enumerate all possible lists of positive integers with even length, sum 30, and mode 9 appearing more times than any other number, with context from Sub-task 0.5 output. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    all_thinking_1_1 = []
    all_answer_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instruction_1_1,
        "context": ["user query", thinking_0_5.content],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_1_1):
        thinking_i, answer_i = await agent([taskInfo, thinking_0_5], debate_instruction_1_1, i, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, enumerating lists, thinking: {thinking_i.content}; answer: {answer_i.content}")
        all_thinking_1_1.append(thinking_i)
        all_answer_1_1.append(answer_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + all_thinking_1_1, "Sub-task 1: Synthesize and choose the most consistent answer for enumerated lists. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing enumerated lists, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = "Sub-task 2: Filter enumerated lists to retain only those whose median is a positive integer not present in the list (i.e., median is the average of two middle elements not in the list), with context from Sub-task 1.1 output."
    N_sc = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, filtering lists by median condition, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings_1_2.append(thinking_i)
        possible_answers_1_2.append(answer_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent filtered lists. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing filtered lists, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    debate_instruction_1_3 = "Sub-task 3: Verify the uniqueness of the mode 9 in the filtered lists, ensuring no other number appears as frequently as 9, with context from Sub-task 1.2 output. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    all_thinking_1_3 = []
    all_answer_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": debate_instruction_1_3,
        "context": ["user query", thinking_1_2.content],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_1_3):
        thinking_i, answer_i = await agent([taskInfo, thinking_1_2], debate_instruction_1_3, i, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, verifying mode uniqueness, thinking: {thinking_i.content}; answer: {answer_i.content}")
        all_thinking_1_3.append(thinking_i)
        all_answer_1_3.append(answer_i)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + all_thinking_1_3, "Sub-task 3: Synthesize and choose the most consistent answer for mode uniqueness verification. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing mode uniqueness verification, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    cot_instruction_2_1 = "Sub-task 1: For the valid list(s) identified, compute the sum of the squares of all the items in the list, with context from Sub-task 1.3 output."
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_3], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, computing sum of squares, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    return final_answer, logs
