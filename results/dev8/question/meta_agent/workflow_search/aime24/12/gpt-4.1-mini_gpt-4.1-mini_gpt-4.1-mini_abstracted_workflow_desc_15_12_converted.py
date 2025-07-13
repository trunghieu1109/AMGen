async def forward_12(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Express the complex number z in polar form with |z|=4, i.e., z=4*e^(i*theta). "
        "Rewrite the expression (75+117i)*z + (96+144i)/z in terms of theta, separating real and imaginary parts. "
        "Compute magnitudes and arguments of coefficients (75+117i) and (96+144i) to facilitate simplification. "
        "Do not assume any specific theta; keep expressions symbolic."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, expressing z in polar form and rewriting expression, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)

    cot_sc_instruction_1_1 = (
        "Sub-task 1.1: Using the polar form and expressions from Sub-task 0, formulate the real part of the expression as a function of theta. "
        "Identify the trigonometric form and simplify it to a single cosine function with amplitude and phase shift if possible. "
        "Keep expressions exact and symbolic, verify domain theta in [0, 2pi]."
    )
    cot_sc_instruction_1_2 = (
        "Sub-task 1.2: Determine the value(s) of theta that maximize the real part function derived in Sub-task 1.1. "
        "Use calculus or trigonometric identities to find critical points and verify maxima. Confirm maximum is achievable within domain."
    )
    N_sc = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]

    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_1, answer_1_1 = await cot_agents_1_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, formulating real part function, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)

    counter_1_1 = Counter([a.content.strip() for a in possible_answers_1_1])
    most_common_answer_1_1 = counter_1_1.most_common(1)[0][0]

    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1(
        [taskInfo] + possible_answers_1_1 + possible_thinkings_1_1,
        "Sub-task 1.1: Synthesize and choose the most consistent real part function for the expression.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "subtask_1_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_2, answer_1_2 = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, finding maximizing theta, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)

    counter_1_2 = Counter([a.content.strip() for a in possible_answers_1_2])
    most_common_answer_1_2 = counter_1_2.most_common(1)[0][0]

    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2(
        [taskInfo] + possible_answers_1_2 + possible_thinkings_1_2,
        "Sub-task 1.2: Synthesize and choose the most consistent maximizing theta value.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    cot_instruction_2_1 = (
        "Sub-task 2.1: Calculate the maximum real part value by substituting the optimal theta found in Sub-task 1.2 into the simplified real part expression from Sub-task 1.1. "
        "Simplify to a final numeric result and present clearly."
    )
    cot_instruction_2_2 = (
        "Sub-task 2.2: Verify the correctness of the maximum real part found by cross-checking with alternative methods, such as geometric interpretation or direct substitution. "
        "Confirm the solution respects |z|=4 and no larger real part is possible. Provide brief justification."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]

    subtask_desc_2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(
        [taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2],
        cot_instruction_2_1,
        is_sub_task=True
    )
    agents.append(f"CoT agent {cot_agent_2_1.id}, calculating max real part value, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "subtask_2_2",
        "instruction": cot_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2_2, answer_2_2 = await cot_agents_2_2[i]([taskInfo, thinking_2_1, answer_2_1], cot_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, verifying max real part correctness, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
        possible_answers_2_2.append(answer_2_2)
        possible_thinkings_2_2.append(thinking_2_2)

    counter_2_2 = Counter([a.content.strip() for a in possible_answers_2_2])
    most_common_answer_2_2 = counter_2_2.most_common(1)[0][0]

    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2(
        [taskInfo] + possible_answers_2_2 + possible_thinkings_2_2,
        "Sub-task 2.2: Synthesize and choose the most consistent verification of max real part.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
