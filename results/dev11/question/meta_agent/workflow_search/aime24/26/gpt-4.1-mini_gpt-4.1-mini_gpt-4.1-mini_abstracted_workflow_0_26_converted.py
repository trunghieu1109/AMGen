async def forward_26(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Formally represent the problem constraints and translate the counting condition into an algebraic expression. "
        "Explain why for each a in A, the number of sets B with max(B) = a is 2^(a-1), and thus sum_{a in A} 2^(a-1) = 2024. "
        "Avoid assuming constraints beyond finiteness and positivity. This sets the foundation for subsequent steps."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing problem constraints, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)

    cot_sc_instruction_2a = (
        "Sub-task 2a: Perform the binary decomposition of 2024 into distinct powers of two with explicit bit indexing starting from LSB as bit 0. "
        "Identify all bits set to 1 in 2024's binary representation and map each set bit at position k to element a = k + 1 in A. "
        "Provide a detailed numeric breakdown showing 2024 as a sum of 2^k terms, explicitly listing powers of two included and corresponding elements in A. "
        "Avoid misindexing bits or including powers exceeding 2024. This step must produce candidate set(s) A exactly satisfying sum_{a in A} 2^(a-1) = 2024."
    )
    N_sc = self.max_sc
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2a = []
    possible_thinkings_2a = []
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_sc_instruction_2a,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2a, answer2a = await cot_agents_2a[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, binary decomposition of 2024, thinking: {thinking2a.content}; answer: {answer2a.content}")
        possible_answers_2a.append(answer2a)
        possible_thinkings_2a.append(thinking2a)

    final_decision_agent_2a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2a, answer2a = await final_decision_agent_2a(
        [taskInfo] + possible_answers_2a + possible_thinkings_2a,
        "Sub-task 2a: Synthesize and choose the most consistent and correct binary decomposition of 2024 into powers of two with explicit bit indexing and mapping to set A.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {"thinking": thinking2a, "answer": answer2a}
    logs.append(subtask_desc2a)

    cot_sc_instruction_2b = (
        "Sub-task 2b: Verify correctness of the candidate set A produced in Sub-task 2a by explicitly computing S = sum_{a in A} 2^{a-1} and confirming S equals 2024. "
        "Reject any candidate sets where equality fails. If multiple candidates exist, select only those passing verification. Document numeric calculations step-by-step. "
        "This enforces correctness before proceeding."
    )
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2b = []
    possible_thinkings_2b = []
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_sc_instruction_2b,
        "context": ["user query", thinking2a.content, answer2a.content],
        "agent_collaboration": "CoT | SC_CoT | Debate | Reflexion"
    }
    for i in range(N_sc):
        thinking2b, answer2b = await cot_agents_2b[i]([taskInfo, thinking2a, answer2a], cot_sc_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, verifying candidate set A correctness, thinking: {thinking2b.content}; answer: {answer2b.content}")
        possible_answers_2b.append(answer2b)
        possible_thinkings_2b.append(thinking2b)

    final_decision_agent_2b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2b, answer2b = await final_decision_agent_2b(
        [taskInfo] + possible_answers_2b + possible_thinkings_2b,
        "Sub-task 2b: Select the verified candidate set A for which sum of 2^{a-1} equals 2024 exactly. Reject others.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {"thinking": thinking2b, "answer": answer2b}
    logs.append(subtask_desc2b)

    cot_instruction_3 = (
        "Stage 2 Sub-task 1: Compute the sum of the elements of the verified set A obtained from Sub-task 2b. "
        "After summing the elements of A, perform a final verification by recalculating sum_{a in A} 2^{a-1} to confirm it equals 2024, ensuring no errors were introduced. "
        "Only after passing this verification should the final sum be reported as the answer. Include all numeric steps and clearly state the final answer and verification result."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking2b.content, answer2b.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2b, answer2b], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, summing elements of verified A and final verification, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
