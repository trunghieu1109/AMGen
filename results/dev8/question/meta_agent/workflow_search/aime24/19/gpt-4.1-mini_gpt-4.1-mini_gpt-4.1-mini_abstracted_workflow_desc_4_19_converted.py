async def forward_19(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Explicitly derive the 13th cyclotomic polynomial Phi_13(x). "
        "Write out Phi_13(x) = 1 + x + x^2 + ... + x^{12}, verify its degree and coefficients, "
        "and confirm it is the minimal polynomial for a primitive 13th root of unity. "
        "Avoid assumptions and provide the explicit polynomial form for subsequent numeric evaluation."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, deriving Phi_13(x), thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_1 = (
        "Sub-task 2: Clarify the nature of omega as a primitive 13th root of unity and confirm the product indexing from k=0 to 12, including omega^0=1. "
        "Rewrite the expression inside the product (2 - 2*omega^k + omega^{2k}) as a polynomial in x = omega^k, simplifying it if possible. "
        "Establish the connection between the product and evaluation of a polynomial at all 13th roots of unity. "
        "Avoid ambiguous interpretations or assumptions about omega or the product indexing."
    )
    N_sc = self.max_sc
    cot_sc_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1, answer_1 = await cot_sc_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, clarifying omega and product indexing, thinking: {thinking_1.content}; answer: {answer_1.content}")
        possible_answers_1.append(answer_1)
        possible_thinkings_1.append(thinking_1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1, answer_1 = await final_decision_agent_1([taskInfo] + possible_answers_1 + possible_thinkings_1, "Sub-task 2: Synthesize and choose the most consistent answer clarifying omega and product indexing.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 3: Evaluate Phi_13(x) explicitly at x = 1 + i and x = 1 - i by substituting term-by-term into the polynomial derived in Sub-task 1. "
        "Compute the sums carefully, separating real and imaginary parts, showing all intermediate steps to avoid errors. "
        "Provide reliable numeric values for Phi_13(1+i) and Phi_13(1-i) for use in resultant calculation."
    )
    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_2, answer_2 = await cot_sc_agents_2[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2[i].id}, evaluating Phi_13 at 1+i and 1-i, thinking: {thinking_2.content}; answer: {answer_2.content}")
        possible_answers_2.append(answer_2)
        possible_thinkings_2.append(thinking_2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2, answer_2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, "Sub-task 3: Synthesize and finalize numeric evaluation of Phi_13(1+i) and Phi_13(1-i).", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 4: Compute the resultant Res(f, Phi_13) = Phi_13(1+i) * Phi_13(1-i) using the numeric values obtained in Sub-task 3. "
        "Provide detailed algebraic and numeric derivation, ensuring consistency and correctness. "
        "Avoid conflicting or unsupported numeric results."
    )
    cot_sc_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc_3 = {
        "subtask_id": "stage_2_subtask_2",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking_1.content, answer_1.content, thinking_2.content, answer_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_3, answer_3 = await cot_sc_agents_3[i]([taskInfo, thinking_1, answer_1, thinking_2, answer_2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3[i].id}, computing resultant Res(f, Phi_13), thinking: {thinking_3.content}; answer: {answer_3.content}")
        possible_answers_3.append(answer_3)
        possible_thinkings_3.append(thinking_3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3, answer_3 = await final_decision_agent_3([taskInfo] + possible_answers_3 + possible_thinkings_3, "Sub-task 4: Synthesize and finalize the numeric value of the resultant.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 4: ", sub_tasks[-1])

    debate_instr_4 = (
        "Sub-task 5: Verify the numeric evaluation of the resultant Res(f, Phi_13) obtained in Sub-task 4. "
        "Use alternative methods such as numeric approximation with higher precision, modular arithmetic checks, or symbolic verification to confirm correctness. "
        "Resolve any conflicts or discrepancies found in previous numeric results and provide a final verified numeric value."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]
    subtask_desc_4 = {
        "subtask_id": "stage_3_subtask_1",
        "instruction": debate_instr_4,
        "context": ["user query", thinking_3.content, answer_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking_4, answer_4 = await agent([taskInfo, thinking_3, answer_3], debate_instr_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking_3, answer_3] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking_4, answer_4 = await agent(input_infos_4, debate_instr_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying resultant, thinking: {thinking_4.content}; answer: {answer_4.content}")
            all_thinking_4[r].append(thinking_4)
            all_answer_4[r].append(answer_4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4, answer_4 = await final_decision_agent_4([taskInfo] + all_thinking_4[-1] + all_answer_4[-1], "Sub-task 5: Synthesize and finalize verified numeric value of the resultant.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 5: ", sub_tasks[-1])

    cot_sc_instruction_5 = (
        "Sub-task 6: Compute the remainder when the verified resultant value from Sub-task 5 is divided by 1000. "
        "Perform modular arithmetic carefully to avoid overflow or errors. "
        "Provide the final remainder as the answer to the original problem. "
        "Reflect on the correctness of the modular reduction and confirm consistency with all previous steps and verifications."
    )
    cot_sc_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_5 = []
    possible_thinkings_5 = []
    subtask_desc_5 = {
        "subtask_id": "stage_3_subtask_2",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", thinking_4.content, answer_4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_5, answer_5 = await cot_sc_agents_5[i]([taskInfo, thinking_4, answer_4], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_5[i].id}, computing remainder modulo 1000, thinking: {thinking_5.content}; answer: {answer_5.content}")
        possible_answers_5.append(answer_5)
        possible_thinkings_5.append(thinking_5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_5, answer_5 = await final_decision_agent_5([taskInfo] + possible_answers_5 + possible_thinkings_5, "Sub-task 6: Synthesize and choose the most consistent remainder modulo 1000.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_5, answer_5, sub_tasks, agents)
    return final_answer, logs
