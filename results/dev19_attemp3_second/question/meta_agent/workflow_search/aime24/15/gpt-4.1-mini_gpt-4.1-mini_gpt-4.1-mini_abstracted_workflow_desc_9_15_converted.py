async def forward_15(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_stage0 = (
        "Sub-task 1: Aggregate and clarify all given numerical data and ownership conditions, "
        "explicitly confirming the role of the universal item (candy hearts) in the counts of exactly two and exactly three items. "
        "Analyze the problem statement carefully and clarify if the counts of exactly two and exactly three items include the candy hearts or not."
    )
    cot_agent_stage0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_stage0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking0, answer0 = await cot_agent_stage0([taskInfo], cot_instruction_stage0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_stage0.id}, clarifying universal item role, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Stage 0 Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {
        "thinking": thinking0,
        "answer": answer0
    }
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])

    debate_instruction_stage1_1 = (
        "Sub-task 1: Determine the distribution of residents owning exactly one, exactly two, exactly three, and exactly four items, "
        "using the aggregated data and clarifications from stage_0.subtask_1. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_stage1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_1 = self.max_round
    all_thinking_1_1 = [[] for _ in range(N_max_1_1)]
    all_answer_1_1 = [[] for _ in range(N_max_1_1)]
    subtask_desc1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instruction_stage1_1,
        "context": ["user query", thinking0.content, answer0.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_1):
        for i, agent in enumerate(debate_agents_stage1_1):
            if r == 0:
                thinking1_1, answer1_1 = await agent([taskInfo, thinking0, answer0], debate_instruction_stage1_1, r, is_sub_task=True)
            else:
                input_infos_1_1 = [taskInfo, thinking0, answer0] + all_thinking_1_1[r-1] + all_answer_1_1[r-1]
                thinking1_1, answer1_1 = await agent(input_infos_1_1, debate_instruction_stage1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, determining distribution of ownership, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
            all_thinking_1_1[r].append(thinking1_1)
            all_answer_1_1[r].append(answer1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_1, answer1_1 = await final_decision_agent_1_1(
        [taskInfo, thinking0, answer0] + all_thinking_1_1[-1] + all_answer_1_1[-1],
        "Sub-task 1: Synthesize and choose the most consistent distribution of ownership." +
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc1_1['response'] = {
        "thinking": thinking1_1,
        "answer": answer1_1
    }
    logs.append(subtask_desc1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_stage1_2 = (
        "Sub-task 2: Enumerate and validate the possible combinations of ownership subsets consistent with the counts of owners of each item "
        "and the counts of residents owning exactly two and exactly three items, using the aggregated data and clarifications from stage_0.subtask_1."
    )
    N_sc = self.max_sc
    cot_agents_stage1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_stage1_2,
        "context": ["user query", thinking0.content, answer0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking1_2, answer1_2 = await cot_agents_stage1_2[i]([taskInfo, thinking0, answer0], cot_sc_instruction_stage1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage1_2[i].id}, enumerating ownership subsets, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
        possible_answers_1_2.append(answer1_2)
        possible_thinkings_1_2.append(thinking1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_2, answer1_2 = await final_decision_agent_1_2(
        [taskInfo, thinking0, answer0] + possible_thinkings_1_2 + possible_answers_1_2,
        "Sub-task 2: Synthesize and choose the most consistent and correct combinations of ownership subsets.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc1_2['response'] = {
        "thinking": thinking1_2,
        "answer": answer1_2
    }
    logs.append(subtask_desc1_2)
    print("Step 1.2: ", sub_tasks[-1])

    debate_instruction_stage2_1 = (
        "Sub-task 1: Infer the number of residents owning all four items by analyzing the relationships between the counts of exactly two and exactly three items, "
        "total owners per item, and the universal ownership of candy hearts, using outputs from stage_1.subtask_1 and stage_1.subtask_2. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_stage2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_stage2_1,
        "context": ["user query", thinking1_1.content, answer1_1.content, thinking1_2.content, answer1_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_stage2_1):
            if r == 0:
                thinking2_1, answer2_1 = await agent([taskInfo, thinking1_1, answer1_1, thinking1_2, answer1_2], debate_instruction_stage2_1, r, is_sub_task=True)
            else:
                input_infos_2_1 = [taskInfo, thinking1_1, answer1_1, thinking1_2, answer1_2] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking2_1, answer2_1 = await agent(input_infos_2_1, debate_instruction_stage2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, inferring number owning all four items, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
            all_thinking_2_1[r].append(thinking2_1)
            all_answer_2_1[r].append(answer2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_1, answer2_1 = await final_decision_agent_2_1(
        [taskInfo, thinking1_1, answer1_1, thinking1_2, answer1_2] + all_thinking_2_1[-1] + all_answer_2_1[-1],
        "Sub-task 1: Synthesize and provide the final answer for the number of residents owning all four items." +
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc2_1['response'] = {
        "thinking": thinking2_1,
        "answer": answer2_1
    }
    logs.append(subtask_desc2_1)
    print("Step 2.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2_1, answer2_1, sub_tasks, agents)
    return final_answer, logs
