async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Formulate the mathematical condition defining b-eautiful numbers for a fixed base b >= 2. "
        "Explicitly state that for digits x,y with 1 <= x <= b-1 and 0 <= y <= b-1, the number n = x*b + y must satisfy n = (x + y)^2, "
        "where sqrt(n) = x + y is an integer. Clarify all digit constraints and assumptions about base representation. "
        "Provide a clear, formal statement of the problem constraints and the key equation to be used in enumeration."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, formulating condition, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)

    cot_sc_instruction_1_1 = (
        "Sub-task 1: For a fixed base b, enumerate all digit pairs (x,y) with 1 <= x <= b-1 and 0 <= y <= b-1. "
        "For each pair, compute n = x*b + y and check if n is a perfect square and if x + y equals sqrt(n). "
        "Explicitly list all valid (x,y,n) triples satisfying the condition. Provide detailed numeric outputs and intermediate results. "
        "Implement pruning but ensure completeness."
    )
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Repeat Sub-task 1 for each base b starting from 2 upwards. "
        "Collect and store all b-eautiful numbers and counts per base in a structured format. "
        "Continue until bases with counts exceeding 10 are found and verified. Provide explicit numeric data for verification."
    )

    N_sc = self.max_sc
    cot_sc_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    cot_sc_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]

    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "SC_CoT"
    }

    for i in range(N_sc):
        thinking_i, answer_i = await cot_sc_agents_1_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_1[i].id}, enumerating digit pairs for given b, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)

    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1(
        [taskInfo] + possible_answers_1_1 + possible_thinkings_1_1,
        "Sub-task 1: Synthesize consistent enumeration of digit pairs for given b.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }

    for i in range(N_sc):
        thinking_i2, answer_i2 = await cot_sc_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_2[i].id}, iterating bases and collecting counts, thinking: {thinking_i2.content}; answer: {answer_i2.content}")
        possible_answers_1_2.append(answer_i2)
        possible_thinkings_1_2.append(thinking_i2)

    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2(
        [taskInfo] + possible_answers_1_2 + possible_thinkings_1_2,
        "Sub-task 2: Synthesize counts of b-eautiful numbers for bases starting from 2.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    debate_instruction_1_3 = (
        "Sub-task 3: Verify and cross-validate the enumerated (x,y,n) triples and counts from previous subtasks. "
        "For each triple, independently confirm n is a perfect square and x + y = sqrt(n). "
        "Check digit constraints and completeness. Log any discrepancies. Provide a verification report confirming correctness and completeness for each base."
    )
    debate_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_3 = self.max_round
    all_thinking_1_3 = [[] for _ in range(N_max_1_3)]
    all_answer_1_3 = [[] for _ in range(N_max_1_3)]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": debate_instruction_1_3,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_1_3):
        for i, agent in enumerate(debate_agents_1_3):
            if r == 0:
                thinking_1_3, answer_1_3 = await agent([taskInfo, thinking_1_2, answer_1_2], debate_instruction_1_3, r, is_sub_task=True)
            else:
                input_infos_1_3 = [taskInfo, thinking_1_2, answer_1_2] + all_thinking_1_3[r-1] + all_answer_1_3[r-1]
                thinking_1_3, answer_1_3 = await agent(input_infos_1_3, debate_instruction_1_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying enumerations, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
            all_thinking_1_3[r].append(thinking_1_3)
            all_answer_1_3[r].append(answer_1_3)

    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3_final, answer_1_3_final = await final_decision_agent_1_3(
        [taskInfo] + all_thinking_1_3[-1] + all_answer_1_3[-1],
        "Sub-task 3: Synthesize verification report confirming correctness and completeness of enumerations.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3_final.content}; answer - {answer_1_3_final.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3_final, "answer": answer_1_3_final}
    logs.append(subtask_desc_1_3)

    cot_reflect_instruction_2_1 = (
        "Sub-task 1: Analyze the verified enumeration data to identify the minimal base b >= 2 for which the count of b-eautiful numbers exceeds 10. "
        "Compare counts across all bases starting from 2 up to the first base exceeding the threshold. "
        "Provide a detailed summary of counts per base, highlight the minimal base meeting the criterion, and confirm minimality by ensuring no smaller base exceeds 10. "
        "Return the minimal base along with verified enumeration counts and a summary of the verification process. Avoid assumptions or extrapolations."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", thinking_1_3_final.content, answer_1_3_final.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2([taskInfo, thinking_1_3_final, answer_1_3_final], cot_reflect_instruction_2_1, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, analyzing verified data and finalizing minimal base, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    for i, step in enumerate(sub_tasks, 1):
        print(f"Step {i}: ", step)
    return final_answer, logs
