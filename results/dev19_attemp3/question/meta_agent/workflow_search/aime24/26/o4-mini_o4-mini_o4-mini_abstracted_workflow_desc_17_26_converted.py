async def forward_26(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: derive total = sum_{m in A} 2^(m-1)
    cot_sc_instruction = (
        "Sub-task 1: Express the total number of finite nonempty subsets B with max(B) in A "
        "as a sum over A and derive that total = sum_{m in A} 2^(m-1)."
    )
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents:
        thinking_i, answer_i = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, deriving formula, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings1.append(thinking_i)
        possible_answers1.append(answer_i)
    final_decision_1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_1(
        [taskInfo] + possible_thinkings1 + possible_answers1,
        "Sub-task 1: Select the most consistent derivation of the expression total = sum_{m in A} 2^(m-1).",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    agents.append(f"Final Decision Agent {final_decision_1.id}, chosen derivation, thinking: {thinking1.content}; answer: {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: decompose 2024 into distinct powers of two
    cot_sc_instruction = (
        "Sub-task 2: Decompose 2024 into a sum of distinct powers of two by converting 2024 to its binary expansion."
    )
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents:
        thinking_i, answer_i = await agent([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, binary decomposition, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings2.append(thinking_i)
        possible_answers2.append(answer_i)
    final_decision_2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_2(
        [taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2,
        "Sub-task 2: Choose the most consistent binary decomposition of 2024 into distinct powers of two.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    agents.append(f"Final Decision Agent {final_decision_2.id}, chosen decomposition, thinking: {thinking2.content}; answer: {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: map powers 2^k to m = k+1 to determine A
    cot_sc_instruction = (
        "Sub-task 3: Map each term 2^k from the decomposition to an element m = k+1, thereby determining the elements of A."
    )
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction,
        "context": ["user query", thinking2, answer2],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents:
        thinking_i, answer_i = await agent([taskInfo, thinking2, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, mapping to A, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings3.append(thinking_i)
        possible_answers3.append(answer_i)
    final_decision_3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_3(
        [taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3,
        "Sub-task 3: Choose the most consistent mapping of powers to elements of A.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    agents.append(f"Final Decision Agent {final_decision_3.id}, chosen A, thinking: {thinking3.content}; answer: {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: compute sum of elements of A
    cot_instruction = (
        "Sub-task 4: Compute the sum of the elements of A as determined in subtask 3."
    )
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction,
        "context": ["user query", thinking3, answer3],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent([taskInfo, thinking3, answer3], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing sum of A, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs