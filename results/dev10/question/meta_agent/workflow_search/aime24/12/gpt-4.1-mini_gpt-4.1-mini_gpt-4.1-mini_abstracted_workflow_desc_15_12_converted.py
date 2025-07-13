async def forward_12(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_0 = (
        "Sub-task 1: Rewrite the given expression (75 + 117i)z + (96 + 144i)/z in terms of the parameter theta, "
        "where z = 4e^{i*theta}. Compute the expression explicitly by substituting z and 1/z, simplifying the complex terms, "
        "and express the real part as a function of theta. Carefully handle the division by z by using the polar form to express 1/z = (1/4)e^{-i*theta}. "
        "Avoid any assumptions beyond the given modulus constraint and ensure the expression is fully reduced to a trigonometric form involving cos(theta) and sin(theta)."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking0, answer0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, rewriting expression in terms of theta, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc_0['response'] = {
        "thinking": thinking0,
        "answer": answer0
    }
    logs.append(subtask_desc_0)
    
    cot_sc_instruction_0 = (
        "Sub-task 1: Based on the output from Sub-task 0, consider/calculate potential cases of rewriting and simplifying the expression further, "
        "ensuring the real part is fully expressed as a trigonometric function of theta. "
        "Use Self-Consistency to verify the correctness and completeness of the trigonometric form."
    )
    N = self.max_sc
    cot_agents_0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_0 = []
    possible_thinkings_0 = []
    subtask_desc_0_sc = {
        "subtask_id": "subtask_1_sc",
        "instruction": cot_sc_instruction_0,
        "context": ["user query", thinking0.content, answer0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking0_sc, answer0_sc = await cot_agents_0[i]([taskInfo, thinking0, answer0], cot_sc_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0[i].id}, verifying trigonometric form, thinking: {thinking0_sc.content}; answer: {answer0_sc.content}")
        possible_answers_0.append(answer0_sc)
        possible_thinkings_0.append(thinking0_sc)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0_sc_final, answer0_sc_final = await final_decision_agent_0([taskInfo] + possible_answers_0 + possible_thinkings_0, 
        "Sub-task 1: Synthesize and choose the most consistent and correct trigonometric form for the real part of the expression.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0 SC output: thinking - {thinking0_sc_final.content}; answer - {answer0_sc_final.content}")
    subtask_desc_0_sc['response'] = {
        "thinking": thinking0_sc_final,
        "answer": answer0_sc_final
    }
    logs.append(subtask_desc_0_sc)
    
    cot_sc_instruction_1 = (
        "Sub-task 2: Identify the values of theta in [0, 2*pi) that maximize the real part of the simplified trigonometric expression obtained in Stage 0. "
        "Analyze the resulting function of theta, possibly rewriting it in the form R*cos(theta - alpha) + S*cos(theta + beta) or combining terms into a single cosine function with a phase shift. "
        "Verify the critical points by differentiation or geometric interpretation, ensuring that the maximum is global on the domain. "
        "Avoid overlooking multiple maxima or boundary conditions."
    )
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking0_sc_final.content, answer0_sc_final.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1, answer1 = await cot_agents_1[i]([taskInfo, thinking0_sc_final, answer0_sc_final], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, finding maximizing theta, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_1.append(answer1)
        possible_thinkings_1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_final, answer1_final = await final_decision_agent_1([taskInfo] + possible_answers_1 + possible_thinkings_1, 
        "Sub-task 2: Synthesize and choose the most consistent and correct maximizing theta values and maximum real part.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_final.content}; answer - {answer1_final.content}")
    subtask_desc_1['response'] = {
        "thinking": thinking1_final,
        "answer": answer1_final
    }
    logs.append(subtask_desc_1)
    
    cot_instruction_2 = (
        "Sub-task 3: Aggregate the maximum value(s) of the real part found in Stage 1 to produce the final largest possible real part of the original expression. "
        "Combine intermediate numeric or symbolic results into a final numeric answer. Verify the correctness of the final value by cross-checking with the expression and constraints. "
        "Present the answer clearly, including the maximum real part and the corresponding theta (or z) value(s) if applicable."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1_final.content, answer1_final.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1_final, answer1_final], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, aggregating final maximum real part, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc_2)
    
    final_answer = await self.make_final_answer(thinking2, answer2, sub_tasks, agents)
    for i, st in enumerate(sub_tasks):
        print(f"Step {i+1}: ", st)
    return final_answer, logs