async def forward_168(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Analyze and classify the original decay process 2A -> 2B + 2E + 2V, "
        "focusing on the nature of emitted particles, energy spectrum characteristics of E particles, "
        "and conservation laws governing the decay."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    N_sc = self.max_sc
    possible_answers_1 = []
    possible_thinkings_1 = []
    for i in range(N_sc):
        thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agent_1.id}, run {i}, analyzing original decay, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    counter_1 = Counter([a.content.strip() for a in possible_answers_1])
    most_common_answer_1 = counter_1.most_common(1)[0][0]
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1(
        [taskInfo] + possible_thinkings_1 + possible_answers_1,
        "Sub-task 1: Synthesize and choose the most consistent and correct analysis of the original decay process.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Analyze and classify the variant decay process 2A -> 2B + 2E + M, "
        "emphasizing the differences introduced by replacing 2V with a single massless particle M and the expected impact on energy distribution and kinematics."
    )
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_2 = []
    possible_thinkings_2 = []
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, run {i}, analyzing variant decay, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    counter_2 = Counter([a.content.strip() for a in possible_answers_2])
    most_common_answer_2 = counter_2.most_common(1)[0][0]
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2(
        [taskInfo, thinking1, answer1] + possible_thinkings_2 + possible_answers_2,
        "Sub-task 2: Synthesize and choose the most consistent and correct analysis of the variant decay process.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_3 = (
        "Sub-task 3: Evaluate and prioritize the possible effects of the variant decay on the energy spectrum of E particles, "
        "particularly regarding continuity versus discreteness and changes in the endpoint energy Q. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instr_3,
        "context": ["user query", thinking1, answer1, thinking2, answer2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking1, answer1, thinking2, answer2], debate_instr_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instr_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating effects on energy spectrum, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_3[r].append(thinking3)
            all_answer_3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3(
        [taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking_3[-1] + all_answer_3[-1],
        "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Compute or reason quantitatively and conditionally how the substitution of 2V by M affects the total energy spectrum shape and endpoint of the outgoing E particles, "
        "applying conservation of energy and momentum and phase space considerations."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3, answer3],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_4 = []
    possible_thinkings_4 = []
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, run {i}, quantitative reasoning on substitution effect, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    counter_4 = Counter([a.content.strip() for a in possible_answers_4])
    most_common_answer_4 = counter_4.most_common(1)[0][0]
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4(
        [taskInfo, thinking3, answer3] + possible_thinkings_4 + possible_answers_4,
        "Sub-task 4: Synthesize and choose the most consistent and correct quantitative reasoning on the substitution effect.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
