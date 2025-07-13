async def forward_11(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0 = (
        "Sub-task 1: Define and characterize the structure of paths on an 8x8 grid from (0,0) to (8,8) "
        "that have exactly four direction changes. Describe the pattern of monotone segments, how direction changes partition the path, "
        "and the implications for the sequence of horizontal and vertical steps. Use detailed reasoning and examples."
    )
    N_sc_0 = self.max_sc
    cot_agents_0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0)]
    possible_answers_0 = []
    possible_thinkings_0 = []
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0):
        thinking0, answer0 = await cot_agents_0[i]([taskInfo], cot_sc_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0[i].id}, defining path structure, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_answers_0.append(answer0)
        possible_thinkings_0.append(thinking0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0(
        [taskInfo] + possible_thinkings_0 + possible_answers_0,
        "Sub-task 1: Synthesize and choose the most consistent and correct characterization of the path structure with exactly four direction changes.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc_0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc_0)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_1_1 = (
        "Sub-task 2: Enumerate all possible distributions of step counts in the five monotone segments "
        "that satisfy the total steps constraints (8 right steps and 8 up steps) and alternate directions starting with either horizontal or vertical. "
        "Use detailed combinatorial reasoning and list all valid segment length tuples."
    )
    N_sc_1_1 = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_1):
        thinking1_1, answer1_1 = await cot_agents_1_1[i]([taskInfo, thinking0, answer0], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, enumerating segment distributions, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
        possible_answers_1_1.append(answer1_1)
        possible_thinkings_1_1.append(thinking1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_1, answer1_1 = await final_decision_agent_1_1(
        [taskInfo, thinking0, answer0] + possible_thinkings_1_1 + possible_answers_1_1,
        "Sub-task 2: Synthesize and choose the most consistent and correct enumeration of valid segment length distributions.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc_1_1)
    print("Step 2.1: ", sub_tasks[-1])

    cot_instruction_1_2 = (
        "Sub-task 3: Calculate the number of distinct paths corresponding to each valid distribution of segment lengths "
        "from Sub-task 2.1 by applying combinatorial formulas (binomial coefficients) for arranging steps within segments. "
        "Provide detailed calculations and final counts for each distribution."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking1_1, answer1_1],
        "agent_collaboration": "CoT"
    }
    thinking1_2, answer1_2 = await cot_agent_1_2([taskInfo, thinking1_1, answer1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, calculating path counts, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking1_2, "answer": answer1_2}
    logs.append(subtask_desc_1_2)
    print("Step 2.2: ", sub_tasks[-1])

    cot_instruction_2_1 = (
        "Sub-task 4: Aggregate the counts from all valid segment length distributions calculated in Sub-task 2.2 "
        "to find the total number of paths with exactly four direction changes. Provide the final total count and reasoning."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking1_2, answer1_2],
        "agent_collaboration": "CoT"
    }
    thinking2_1, answer2_1 = await cot_agent_2_1([taskInfo, thinking1_2, answer1_2], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, aggregating counts, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc_2_1)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2_1, answer2_1, sub_tasks, agents)
    return final_answer, logs
