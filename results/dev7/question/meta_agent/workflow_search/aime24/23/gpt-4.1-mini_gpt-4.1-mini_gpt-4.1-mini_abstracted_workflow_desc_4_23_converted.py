async def forward_23(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Derive formal numeric representations for the digits in the 2x3 grid. "
        "Define variables a,b,c,d,e,f as digits 0-9 for each cell. Express the two row numbers as R1=100a+10b+c and R2=100d+10e+f, "
        "and the three column numbers as C1=10a+d, C2=10b+e, C3=10c+f. Formulate the sum constraints: R1+R2=999 and C1+C2+C3=99. "
        "Validate assumptions such as allowing leading zeros and digit repetition."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, deriving numeric representations, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)

    cot_sc_instruction_1 = (
        "Sub-task 1: Identify and enumerate all digit assignments (a,b,c,d,e,f) from 0 to 9 satisfying the row sum constraint R1+R2=999. "
        "Iterate or simplify algebraically to reduce search space. Verify digit bounds and sum condition."
    )
    N_sc = self.max_sc
    cot_sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", f"thinking of subtask 1", f"answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1, answer_1 = await cot_sc_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, enumerating digit assignments for row sum, thinking: {thinking_1.content}; answer: {answer_1.content}")
        possible_answers_1.append(answer_1)
        possible_thinkings_1.append(thinking_1)

    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1, answer_1 = await final_decision_agent_1(
        [taskInfo] + possible_answers_1 + possible_thinkings_1,
        "Sub-task 1: Synthesize and choose the most consistent digit assignments satisfying R1+R2=999.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)

    debate_instruction_2 = (
        "Sub-task 1: From candidate digit assignments satisfying R1+R2=999, filter those also satisfying C1+C2+C3=99. "
        "Analyze numeric relationships and simplify constraints to reduce complexity. Use debate and reflexion to identify patterns and invariants."
    )
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds_2 = self.max_round
    all_thinking_2 = [[] for _ in range(N_rounds_2)]
    all_answer_2 = [[] for _ in range(N_rounds_2)]
    subtask_desc_2 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instruction_2,
        "context": ["user query", thinking_1, answer_1],
        "agent_collaboration": "Debate"
    }
    for r in range(N_rounds_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking_2, answer_2 = await agent([taskInfo, thinking_1, answer_1], debate_instruction_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking_1, answer_1] + all_thinking_2[r-1] + all_answer_2[r-1]
                thinking_2, answer_2 = await agent(input_infos_2, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, filtering digit assignments for column sum, thinking: {thinking_2.content}; answer: {answer_2.content}")
            all_thinking_2[r].append(thinking_2)
            all_answer_2[r].append(answer_2)

    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2, answer_2 = await final_decision_agent_2(
        [taskInfo] + all_thinking_2[-1] + all_answer_2[-1],
        "Sub-task 2: Synthesize and finalize digit assignments satisfying both sum constraints.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)

    cot_instruction_3 = (
        "Sub-task 1: Aggregate all valid digit assignments found after applying constraints and simplifications. "
        "Compute the total number of ways to fill the 2x3 grid satisfying both sum conditions. "
        "Verify completeness and correctness of the count."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking_2, answer_2],
        "agent_collaboration": "CoT"
    }
    thinking_3, answer_3 = await cot_agent_3([taskInfo, thinking_2, answer_2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, aggregating valid assignments and computing total count, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)

    final_answer = await self.make_final_answer(thinking_3, answer_3, sub_tasks, agents)
    return final_answer, logs
