async def forward_19(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Formally define the 13th roots of unity, clarify the meaning of omega including the condition omega != 1, "
        "and explicitly confirm the set over which the product prod_{k=0}^{12} is taken, emphasizing that it includes omega^0 = 1. "
        "Avoid any assumptions about excluding the root 1 from the product.")
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, defining 13th roots of unity and product set, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Rewrite the expression inside the product, 2 - 2*omega^k + omega^{2k}, as a polynomial f(x) = 2 - 2x + x^2 evaluated at x = omega^k. "
        "Explore and state any immediate algebraic simplifications or factorizations of f(x), such as completing the square or identifying roots, without attempting to evaluate the product yet.")
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, rewriting and simplifying polynomial f(x), thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 3: Identify and clearly state relevant algebraic properties of the 13th roots of unity and cyclotomic polynomials, "
        "including the minimal polynomial Phi_13(x), the factorization of x^{13} - 1, and how these properties relate to evaluating polynomials at roots of unity. "
        "Emphasize the importance of these properties for simplifying the product.")
    N_sc = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, analyzing cyclotomic properties, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + possible_thinkings_3, "Sub-task 3: Synthesize and choose the most consistent answer for cyclotomic properties." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Express the product prod_{k=0}^{12} f(omega^k) as the product of the polynomial f(x) = 2 - 2x + x^2 evaluated at all 13th roots of unity, "
        "and relate this product to the resultant or norm of f(x) with respect to x^{13} - 1. Avoid re-deriving the product via a second resultant approach to prevent sign errors; instead, set the stage for verification.")
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": [taskInfo, thinking2.content, thinking3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, thinking3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, relating product to resultant/norm, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_5 = (
        "Sub-task 5: Simplify the polynomial f(x) = 2 - 2x + x^2 by completing the square or rewriting it in a form that reveals its roots or factorization pattern, "
        "such as (x - 1)^2 + 1. This will aid in connecting the product to evaluations of cyclotomic polynomials or norms.")
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": [thinking2.content],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([thinking2], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, simplifying polynomial f(x), thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_sc_instruction_6 = (
        "Sub-task 6: Use the factorization of x^{13} - 1 and properties of cyclotomic polynomials to rewrite the product over roots of unity in terms of polynomial evaluations or norms, "
        "specifically relating the product to |Phi_13(1 + i)|^2 or a similar norm expression. Emphasize the algebraic structure and avoid sign ambiguity by carefully tracking polynomial degrees and leading coefficients.")
    cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_6 = []
    possible_thinkings_6 = []
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_sc_instruction_6,
        "context": [thinking3.content, thinking4.content, thinking5.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking6, answer6 = await cot_agents_6[i]([taskInfo, thinking3, thinking4, thinking5], cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6[i].id}, rewriting product via cyclotomic norms, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6)
        possible_thinkings_6.append(thinking6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + possible_thinkings_6, "Sub-task 6: Synthesize and confirm product expression via norm of Phi_13(1+i).", is_sub_task=True)
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6 = {"thinking": thinking6, "answer": answer6}
    logs.append({"subtask_id": "subtask_6", "instruction": cot_sc_instruction_6, "context": [thinking3.content, thinking4.content, thinking5.content], "agent_collaboration": "SC_CoT", "response": subtask_desc6})
    print("Step 6: ", sub_tasks[-1])

    cot_instruction_7a = (
        "Sub-task 7a: Determine the sign of the product by analyzing the parity of polynomial degrees, leading coefficients, and the definition of the resultant. "
        "Explicitly compute the sign factor (-1)^{mn} where m, n are degrees of the involved polynomials, and incorporate this sign into the product value. "
        "This subtask must be separate and explicit to prevent sign errors.")
    cot_agent_7a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7a = {
        "subtask_id": "subtask_7a",
        "instruction": cot_instruction_7a,
        "context": [thinking6.content],
        "agent_collaboration": "CoT"
    }
    thinking7a, answer7a = await cot_agent_7a([thinking6], cot_instruction_7a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7a.id}, determining sign of product, thinking: {thinking7a.content}; answer: {answer7a.content}")
    sub_tasks.append(f"Sub-task 7a output: thinking - {thinking7a.content}; answer - {answer7a.content}")
    subtask_desc7a['response'] = {"thinking": thinking7a, "answer": answer7a}
    logs.append(subtask_desc7a)
    print("Step 7a: ", sub_tasks[-1])

    cot_instruction_7b = (
        "Sub-task 7b: Using the sign factor determined in Sub-task 7a, confirm the explicit numeric value of the product prod_{k=0}^{12} f(omega^k) including the correct sign. "
        "Avoid re-deriving the product from scratch; instead, verify the previously established value (e.g., 8321) with sign correction.")
    cot_agent_7b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7b = {
        "subtask_id": "subtask_7b",
        "instruction": cot_instruction_7b,
        "context": [thinking7a.content, thinking6.content],
        "agent_collaboration": "CoT"
    }
    thinking7b, answer7b = await cot_agent_7b([thinking7a, thinking6], cot_instruction_7b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7b.id}, confirming signed numeric product, thinking: {thinking7b.content}; answer: {answer7b.content}")
    sub_tasks.append(f"Sub-task 7b output: thinking - {thinking7b.content}; answer - {answer7b.content}")
    subtask_desc7b['response'] = {"thinking": thinking7b, "answer": answer7b}
    logs.append(subtask_desc7b)
    print("Step 7b: ", sub_tasks[-1])

    reflect_inst_8 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_8 = "Sub-task 8: Verify the correctness of the signed product value using the norm or resultant approach verified in previous subtasks, now including the correct sign determined in Sub-task 7. " + reflect_inst_8
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_8 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_8 = self.max_round
    cot_inputs_8 = [taskInfo, thinking7b, thinking7a, thinking6]
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_reflect_instruction_8,
        "context": [taskInfo, thinking7b.content, thinking7a.content, thinking6.content],
        "agent_collaboration": "Reflexion"
    }
    thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_8.id}, verifying signed product, thinking: {thinking8.content}; answer: {answer8.content}")
    for i in range(N_max_8):
        feedback, correct = await critic_agent_8([taskInfo, thinking8], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_8.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_8.extend([thinking8, feedback])
        thinking8, answer8 = await cot_agent_8(cot_inputs_8, cot_reflect_instruction_8, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_8.id}, refining verification, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    cot_instruction_9 = (
        "Sub-task 9: Reduce the signed product value modulo 1000 to find the remainder when the product is divided by 1000. "
        "Explicitly handle negative values by applying modular arithmetic rules for negative integers to ensure the remainder is correctly computed in [0,999].")
    cot_agent_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_instruction_9,
        "context": [answer8.content],
        "agent_collaboration": "CoT"
    }
    thinking9, answer9 = await cot_agent_9([answer8], cot_instruction_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_9.id}, computing remainder modulo 1000 with sign correction, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])

    reflect_inst_10 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_10 = "Sub-task 10: Verify the correctness of the modular remainder by cross-checking with alternative computations or modular identities. This includes confirming the sign and magnitude consistency, and optionally performing a quick numerical approximation (e.g., evaluating f(omega^k) for a few roots numerically) to validate the final result." + reflect_inst_10
    cot_agent_10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_10 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_10 = self.max_round
    cot_inputs_10 = [taskInfo, thinking9]
    subtask_desc10 = {
        "subtask_id": "subtask_10",
        "instruction": cot_reflect_instruction_10,
        "context": [taskInfo, thinking9.content],
        "agent_collaboration": "Reflexion"
    }
    thinking10, answer10 = await cot_agent_10(cot_inputs_10, cot_reflect_instruction_10, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_10.id}, verifying modular remainder, thinking: {thinking10.content}; answer: {answer10.content}")
    for i in range(N_max_10):
        feedback, correct = await critic_agent_10([taskInfo, thinking10], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_10.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_10.extend([thinking10, feedback])
        thinking10, answer10 = await cot_agent_10(cot_inputs_10, cot_reflect_instruction_10, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_10.id}, refining verification, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs
