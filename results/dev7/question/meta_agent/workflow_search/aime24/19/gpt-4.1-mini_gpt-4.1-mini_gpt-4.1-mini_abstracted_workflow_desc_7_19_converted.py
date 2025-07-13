async def forward_19(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Derive a simplified algebraic form of the expression inside the product, "
        "i.e., rewrite (2 - 2*omega^k + omega^{2k}) in a more tractable form. "
        "Explore factorization or polynomial identities that relate this expression to known polynomials or roots of unity properties. "
        "Validate that the expression can be interpreted as a polynomial evaluated at omega^k, and confirm the inclusion of k=0 (omega^0=1) term. "
        "Avoid assumptions that omega is primitive unless explicitly needed; just use omega^{13} = 1. "
        "This subtask sets the foundation for further analysis by establishing a clear, simplified representation of the product terms."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing expression inside product, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)

    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, establish the connection between the product over k of the expression evaluated at omega^k "
        "and the resultant or norm of a polynomial related to the 13th cyclotomic polynomial. Identify or construct a polynomial P(x) such that the product equals the product of P(omega^k) over all 13th roots of unity. "
        "Use properties of cyclotomic polynomials and roots of unity to express the product as a polynomial resultant or norm, enabling algebraic simplification. "
        "This step requires careful algebraic reasoning and validation of polynomial identities."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, establishing polynomial connection, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2, 
                                                    "Sub-task 2: Synthesize and choose the most consistent and correct polynomial representation for the product.", 
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)

    cot_sc_instruction_3 = (
        "Sub-task 3: Select and verify the polynomial P(x) identified in Stage 0 that corresponds to the expression inside the product. "
        "Confirm that P(x) is well-defined over the field containing the 13th roots of unity and that its degree and coefficients are consistent with the problem. "
        "Enumerate the roots of unity and verify that evaluating P(x) at these roots matches the original expression. "
        "Check for any roots or factors that simplify the product, such as repeated roots or roots at unity. "
        "This step ensures the correctness and applicability of the polynomial representation before aggregation."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, verifying polynomial P(x), thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + possible_answers_3 + possible_thinkings_3, 
                                                    "Sub-task 3: Synthesize and confirm the correctness of polynomial P(x) representation.", 
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)

    reflexion_instruction_4 = (
        "Sub-task 4: Derive the composite measure: compute the product of P(omega^k) over all k=0 to 12 using the polynomial representation and properties of roots of unity. "
        "Use algebraic identities, such as evaluating P(x^{13}) or leveraging the factorization of x^{13}-1, to find a closed-form expression or integer value for the product. "
        "Then, reduce the resulting integer modulo 1000 to find the remainder. "
        "This step combines all previous results and requires careful modular arithmetic and verification of the final answer."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": reflexion_instruction_4,
        "context": ["user query", "thinking and answers of subtasks 1,2,3"],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, reflexion_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, computing product and modulo, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], 
                                                "Please review and provide the limitations of provided solutions. If correct, output exactly 'True' in 'correct'.", 
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, reflexion_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining computation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)

    reflexion_instruction_5 = (
        "Sub-task 5: Verify the final computed remainder modulo 1000 by cross-checking with alternative methods or partial computations (e.g., numerical approximations, modular arithmetic checks). "
        "Confirm the consistency and correctness of the final answer. Provide a clear statement of the final remainder and document the verification process."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_5 = [taskInfo, thinking4, answer4]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": reflexion_instruction_5,
        "context": ["user query", "thinking and answer of subtask 4"],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflexion_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, verifying final remainder, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], 
                                                "Please review and provide the limitations of provided verification. If correct, output exactly 'True' in 'correct'.", 
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflexion_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining verification, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    for i, step in enumerate(sub_tasks, 1):
        print(f"Step {i}: ", step)
    return final_answer, logs
