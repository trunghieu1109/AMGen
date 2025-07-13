async def forward_12(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    cot_instruction_0_1 = "Sub-task 1: Formally represent the complex number z with the constraint |z|=4 using polar form z = 4e^{iθ}, where θ ∈ [0, 2π). Emphasize that θ is the only variable parameter and that the magnitude constraint fixes the radius. Avoid introducing any assumptions about the expression or its optimization at this stage."
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, stage_0.subtask_1, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"stage_0.subtask_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = "Sub-task 2: Express the given complex expression (75 + 117i)z + (96 + 144i)/z in terms of θ by substituting z = 4e^{iθ} and simplifying the expression accordingly. Carefully handle the reciprocal term by incorporating the magnitude constraint |z|=4 explicitly. Do not attempt optimization yet; focus solely on correct algebraic and trigonometric rewriting."
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent([taskInfo, thinking_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, stage_0.subtask_2, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"stage_0.subtask_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    cot_instruction_0_3 = "Sub-task 3: Identify and isolate the real part of the expression obtained in subtask_2 as a function of θ. Clearly separate the real and imaginary components without performing any optimization or simplification beyond this extraction. Avoid assumptions about the form or maximum value at this stage."
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent([taskInfo, thinking_0_2], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, stage_0.subtask_3, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"stage_0.subtask_3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    N_sc = self.max_sc
    cot_sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    cot_sc_instruction_1 = "Sub-task 1: Rewrite the real part function from stage_0 subtask_3 into a trigonometric form involving cos(θ) and sin(θ), combining like terms and constants to simplify the expression for easier analysis. Ensure all coefficients are correctly computed, explicitly reflecting the contribution of both z and 1/z terms."
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking_0_3.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    for i in range(N_sc):
        thinking_1_1, answer_1_1 = await cot_sc_agents_1[i]([taskInfo, thinking_0_3], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, stage_1.subtask_1, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_thinkings_1_1, "Sub-task 1: Synthesize and choose the most consistent answer for rewriting real part into trig form.", is_sub_task=True)
    sub_tasks.append(f"stage_1.subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    cot_sc_instruction_1_2 = "Sub-task 2: Transform the trigonometric expression from subtask_1 into a single cosine function with a phase shift, i.e., rewrite A cos(θ) + B sin(θ) as R cos(θ - φ), and determine R and φ explicitly. Provide clear formulas and intermediate steps to avoid ambiguity. Do not perform maximum value calculation here."
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    for i in range(N_sc):
        thinking_1_2, answer_1_2 = await cot_sc_agents_1_2[i]([taskInfo, thinking_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_2[i].id}, stage_1.subtask_2, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent answer for rewriting trig expression as single cosine.", is_sub_task=True)
    sub_tasks.append(f"stage_1.subtask_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    debate_instr_1_3 = "Sub-task 3: Rigorously analyze and verify the maximum real part of the expression by considering the full structure including the reciprocal term and the fixed magnitude constraint. Reformulate the expression if needed (e.g., using z and its conjugate or other complex analysis tools). Apply mathematical inequalities or bounding techniques such as the Cauchy-Schwarz inequality to determine whether the amplitude R found previously is attainable or if constraints reduce it. Provide a justified numeric maximum real part value consistent with the original problem. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_1_3 = "Sub-task 3: Your problem is to rigorously verify the maximum real part of the expression considering the full complex structure and constraints." + debate_instr_1_3
    debate_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_3 = self.max_round
    all_thinking_1_3 = [[] for _ in range(N_max_1_3)]
    all_answer_1_3 = [[] for _ in range(N_max_1_3)]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": debate_instruction_1_3,
        "context": ["user query", thinking_1_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_3):
        for i, agent in enumerate(debate_agents_1_3):
            if r == 0:
                thinking_1_3, answer_1_3 = await agent([taskInfo, thinking_1_2], debate_instruction_1_3, r, is_sub_task=True)
            else:
                input_infos_1_3 = [taskInfo, thinking_1_2] + all_thinking_1_3[r-1]
                thinking_1_3, answer_1_3 = await agent(input_infos_1_3, debate_instruction_1_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, stage_1.subtask_3, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
            all_thinking_1_3[r].append(thinking_1_3)
            all_answer_1_3[r].append(answer_1_3)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + all_thinking_1_3[-1], "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final numeric maximum real part.", is_sub_task=True)
    sub_tasks.append(f"stage_1.subtask_3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    cot_instruction_2_1 = "Sub-task 1: Compute the numeric maximum real part value of the expression based on the analysis from stage_1 subtask_3. Explicitly calculate the maximum value as an integer, ensuring all intermediate results are consistent and correctly rounded if necessary. Avoid outputting any angle or intermediate parameters; focus solely on the numeric maximum real part."
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent([taskInfo, thinking_1_3], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, stage_2.subtask_1, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"stage_2.subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    reflect_inst_2_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_2 = "Sub-task 2: Verify the computed maximum real part by substituting the corresponding θ value back into the original expression and confirming that the magnitude constraint |z|=4 is satisfied. Validate that the maximum value is achievable and consistent with the original problem statement. Document any bounding arguments or checks performed to ensure correctness. Avoid re-deriving the expression; focus on validation only." + reflect_inst_2_2
    cot_agent_reflect = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_reflect = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, thinking_1_3]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content, thinking_1_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_reflect(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_reflect.id}, stage_2.subtask_2, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_reflect):
        feedback, correct = await critic_agent([taskInfo, thinking_2_2], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, stage_2.subtask_2, feedback: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, feedback])
        thinking_2_2, answer_2_2 = await cot_agent_reflect(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_reflect.id}, stage_2.subtask_2, refining, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"stage_2.subtask_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    cot_instruction_2_3 = "Sub-task 3: Output the final answer as the largest possible real part of the given expression, formatted as an integer as requested by the user. Ensure that this output is clearly distinguished from any intermediate values or parameters. Confirm that the output aligns with the verified maximum from previous subtasks."
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_instruction_2_3,
        "context": ["user query", thinking_2_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_3, answer_2_3 = await cot_agent([taskInfo, thinking_2_2], cot_instruction_2_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, stage_2.subtask_3, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"stage_2.subtask_3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 2.3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_3, answer_2_3, sub_tasks, agents)
    return final_answer, logs
