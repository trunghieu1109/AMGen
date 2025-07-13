async def forward_19(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Rewrite and simplify the expression inside the product: analyze the quadratic expression "
        "2 - 2*omega^k + omega^(2k) to find an equivalent simpler form suitable for relating the product to a polynomial evaluated at the 13th roots of unity. "
        "Use algebraic identities and properties of roots of unity, such as rewriting the expression in a factored or transformed form. "
        "Avoid direct numerical expansion or brute force computation. Explicitly justify each algebraic step and avoid unstated assumptions."
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
        "Sub-task 2: Based on the output from Sub-task 1, express the entire product \(\prod_{k=0}^{12} (2 - 2\omega^k + \omega^{2k})\) "
        "as a polynomial evaluated at the 13th roots of unity or as a resultant of related polynomials. Identify or construct a polynomial whose roots are \(\omega^k\) and relate the product to its value at a specific point or to the resultant of polynomials. "
        "Use properties of cyclotomic polynomials and the primitive nature of \(\omega\). Provide full algebraic justification for all steps. Avoid skipping steps or relying on heuristics without proof."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1.content, answer1.content], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, expressing product as polynomial, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + [a.content for a in possible_answers_2] + [t.content for t in possible_thinkings_2],
                                                    "Sub-task 2: Synthesize and choose the most consistent and correct expression for the product as a polynomial.",
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)

    cot_sc_instruction_3 = (
        "Sub-task 3: Compute or infer the exact numeric value of the product from the polynomial representation found in stage 1. "
        "Evaluate the polynomial or resultant explicitly, simplify the expression using algebraic identities, and possibly use known values of cyclotomic polynomials at specific points. "
        "Perform careful algebraic manipulation to avoid errors. Prepare the exact numeric result as a single integer value to be used in subsequent modular reduction. Document the derivation clearly and avoid approximations."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2.content, answer2.content], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, computing exact numeric value, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + [a.content for a in possible_answers_3] + [t.content for t in possible_thinkings_3],
                                                    "Sub-task 3: Synthesize and choose the most consistent and correct exact numeric value for the product.",
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)

    numeric_value = None
    try:
        numeric_value = int(''.join(filter(str.isdigit, answer3.content)))
    except Exception:
        numeric_value = None

    cot_instruction_4 = (
        "Sub-task 4: Given the exact numeric value of the product from Sub-task 3, perform a deterministic modular reduction by 1000. "
        "Explicitly compute the remainder of the numeric value modulo 1000 and output only that remainder as the final answer. "
        "Avoid any free-form reflection or heuristic reasoning to ensure consistency and correctness. "
        "Use the numeric value from Sub-task 3 as direct input."
    )
    class NumericReductionAgent:
        async def __call__(self, inputs, instruction, is_sub_task=False):
            val = None
            for inp in inputs:
                try:
                    val = int(''.join(filter(str.isdigit, inp)))
                    break
                except:
                    continue
            if val is None:
                return ("", "Error: No valid numeric value found for modular reduction.")
            remainder = val % 1000
            return ("", str(remainder))

    numeric_reduction_agent = NumericReductionAgent()
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", answer3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await numeric_reduction_agent([answer3.content], cot_instruction_4, is_sub_task=True)
    agents.append(f"Numeric Reduction agent, performing modulo 1000 on numeric value, answer: {answer4}")
    sub_tasks.append(f"Sub-task 4 output: answer - {answer4}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)

    reflect_inst_5 = (
        "Sub-task 5: Verify the entire solution's consistency and correctness by cross-checking: "
        "(a) that the modular reduction in Sub-task 4 matches the numeric value from Sub-task 3 modulo 1000, "
        "(b) that the algebraic simplifications in Subtasks 1 and 2 logically lead to the numeric evaluation in Sub-task 3, "
        "and (c) that no assumptions or errors have propagated. Provide a final answer alongside a verification statement (True if consistent, else detailed discrepancy). "
        "This subtask isolates verification from computation to reduce error propagation and enforces a strict chain of trust in the outputs."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Reflexion Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": reflect_inst_5,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content, thinking3.content, answer3.content, answer4],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_5 = [taskInfo, thinking1.content, answer1.content, thinking2.content, answer2.content, thinking3.content, answer3.content, answer4]
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflect_inst_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent, verifying solution consistency, thinking: {thinking5.content}; answer: {answer5.content}")
    critic_inst_5 = (
        "Please review the answer above and criticize on where might be wrong. "
        "Explicitly check if the modular reduction in Sub-task 4 matches the numeric value from Sub-task 3 modulo 1000. "
        "If consistent, output exactly 'True' in 'correct'. Otherwise, provide detailed discrepancy."
    )
    for i in range(self.max_round):
        feedback, correct = await critic_agent_5(cot_inputs_5 + [thinking5.content, answer5.content],
                                                critic_inst_5, i, is_sub_task=True)
        agents.append(f"Critic agent, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_5.extend([thinking5.content, answer5.content, feedback.content])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflect_inst_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent, refining verification, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
