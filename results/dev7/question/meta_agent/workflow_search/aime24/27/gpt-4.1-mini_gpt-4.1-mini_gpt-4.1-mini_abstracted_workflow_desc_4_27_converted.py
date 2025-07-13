async def forward_27(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Derive formal modular arithmetic representations for the problem constraints. "
        "Represent N as a four-digit number with digits a,b,c,d and express the condition that changing any one digit to 1 results in a number divisible by 7. "
        "Formulate these divisibility conditions as modular equations involving a,b,c,d. "
        "Validate assumptions such as the number remaining four-digit after digit changes and clarify the domain of digits (0-9). "
        "Avoid assuming multiple digits change simultaneously or leading zeros after digit change."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, deriving modular equations, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Relate and validate the modular equations derived in Sub-task 1 to ensure consistency and feasibility. "
        "Confirm that the system of modular equations correctly models the problem constraints and that the domain restrictions on digits are properly incorporated. "
        "Prepare these equations for use in enumeration or algebraic solving in the next stage."
    )
    N_sc = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, validating modular equations, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2(
        [taskInfo] + possible_answers_0_2 + possible_thinkings_0_2,
        "Sub-task 2: Synthesize and choose the most consistent and correct modular equations for the problem.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Identify all four-digit numbers N = a b c d that satisfy the modular conditions derived in Stage 0. "
        "Enumerate or algebraically solve the system of modular equations to find all digit quadruples (a,b,c,d) such that changing any one digit to 1 yields a multiple of 7. "
        "Among these, select the greatest such number N. Carefully consider digit constraints and ensure no invalid numbers (e.g., leading zero) are included. "
        "Avoid brute force without modular reasoning to reduce search space."
    )
    N_sc_1_1 = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", "thinking of subtask 2 stage 0", "answer of subtask 2 stage 0"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_1):
        thinking_i, answer_i = await cot_agents_1_1[i]([taskInfo, thinking_0_2, answer_0_2], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, enumerating candidates, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1(
        [taskInfo] + possible_answers_1_1 + possible_thinkings_1_1,
        "Sub-task 1: Select the greatest valid N satisfying all modular conditions.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Verify the uniqueness and maximality of the identified N. "
        "Confirm that no larger four-digit number satisfies the conditions and that the found N indeed meets the divisibility property for all digit changes. "
        "This verification should include checking all digit-change cases explicitly."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", "thinking of subtask 1 stage 1", "answer of subtask 1 stage 1"],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1, answer_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, verifying maximality, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    debate_instruction_2_1 = (
        "Sub-task 1: Decompose the identified number N into quotient Q and remainder R upon division by 1000, i.e., compute Q = floor(N/1000) and R = N mod 1000. "
        "Simplify these components if applicable and prepare them for summation. Ensure correct integer division and remainder extraction."
    )
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", "thinking of subtask 2 stage 1", "answer of subtask 2 stage 1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_2, answer_1_2], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_2, answer_1_2] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, decomposing N, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_2_1[r].append(thinking_i)
            all_answer_2_1[r].append(answer_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1(
        [taskInfo] + all_thinking_2_1[-1] + all_answer_2_1[-1],
        "Sub-task 1: Finalize decomposition of N into Q and R.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    reflect_instruction_2_2 = (
        "Sub-task 2: Compute the sum Q + R as required by the problem. "
        "Confirm the arithmetic correctness and prepare the final numeric result for output."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "subtask_2",
        "instruction": reflect_instruction_2_2,
        "context": ["user query", "thinking of subtask 2 stage 2 subtask 1", "answer of subtask 2 stage 2 subtask 1"],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, computing Q+R, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback, correct = await critic_agent_2_2([taskInfo, thinking_2_2, answer_2_2],
                                                  "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, answer_2_2, feedback])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining Q+R computation, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    cot_instruction_3_1 = (
        "Sub-task 1: Aggregate and combine all intermediate results to produce the final answer. "
        "This includes consolidating the verified maximal N, the decomposition into Q and R, and the computed sum Q + R. "
        "Present the final answer clearly and verify that all problem conditions have been met."
    )
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    N_sc_3_1 = self.max_sc
    cot_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3_1)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_3_1,
        "context": ["user query", "thinking of subtask 1 stage 0", "thinking of subtask 2 stage 1", "thinking of subtask 2 stage 2", "answer of subtask 2 stage 2"],
        "agent_collaboration": "CoT"
    }
    for i in range(N_sc_3_1):
        thinking_i, answer_i = await cot_agents_3_1[i]([
            taskInfo, thinking_0_1, answer_0_1, thinking_1_2, answer_1_2, thinking_2_2, answer_2_2
        ], cot_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT agent {cot_agents_3_1[i].id}, aggregating final answer, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_3_1.append(answer_i)
        possible_thinkings_3_1.append(thinking_i)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1(
        [taskInfo] + possible_answers_3_1 + possible_thinkings_3_1,
        "Sub-task 1: Provide the final consolidated answer Q+R.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
