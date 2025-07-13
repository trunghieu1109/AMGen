async def forward_19(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Define roots and express product as polynomial evaluations
    cot_instruction_0_1 = (
        "Sub-task 1: Formally define the 13th roots of unity. Explicitly state that omega is a primitive 13th root of unity satisfying omega^13 = 1, "
        "and clarify that the product runs over all roots omega^k for k=0 to 12, including omega^0=1. Avoid ambiguity about excluding omega=1 from the product."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, defining 13th roots of unity, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Express the product \u220f_{k=0}^{12} (2 - 2*omega^k + omega^{2k}) as the product of evaluations of the polynomial f(x) = 2 - 2x + x^2 at all 13th roots of unity. "
        "Emphasize the interpretation of the product as \u220f_{omega^{13}=1} f(omega)."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, expressing product as polynomial evaluations, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    cot_sc_instruction_0_3 = (
        "Sub-task 3: Identify and state algebraic properties of f(x) = 2 - 2x + x^2, including degree and rewrite it in simpler form such as (x-1)^2 + 1. "
        "Explain how this facilitates evaluation at roots of unity and reduces complexity."
    )
    N_sc = self.max_sc
    cot_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_3[i]([taskInfo, thinking_0_2, answer_0_2], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_3[i].id}, simplifying polynomial f(x), thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_3.append(answer_i)
        possible_thinkings_0_3.append(thinking_i)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_answers_0_3 + possible_thinkings_0_3, "Sub-task 3: Synthesize and choose the most consistent and correct simplification of f(x)", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_0_3.id}, synthesizing polynomial simplification, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)

    # Stage 1: Algebraic validation and resultant formula application
    cot_reflect_instruction_1_1 = (
        "Sub-task 1: Verify the factorization and simplification of f(x) by evaluating it at test points such as x=1 and x=i. "
        "Confirm sign patterns and correctness of f(x) = (x-1)^2 + 1. This serves as algebraic validation and sanity check."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_reflect_instruction_1_1,
        "context": ["user query", thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_3, answer_0_3], cot_reflect_instruction_1_1, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, verifying polynomial factorization, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    critic_inst_1_1 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback_1_1, correct_1_1 = await critic_agent_1_1([taskInfo, thinking_1_1, answer_1_1], critic_inst_1_1, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_1.id}, providing feedback, thinking: {feedback_1_1.content}; answer: {correct_1_1.content}")
        if correct_1_1.content == "True":
            break
        thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_3, answer_0_3, thinking_1_1, answer_1_1, feedback_1_1], cot_reflect_instruction_1_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, refining verification, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    cot_reflect_instruction_1_2 = (
        "Sub-task 2: Recall and explicitly state the resultant formula for polynomials f(x) and g(x) = x^{13} - 1, including degrees and leading coefficients. "
        "Emphasize correct exponentiation: raise f(alpha) to degree of g and g(beta) to degree of f, where alpha, beta are roots of g and f respectively. Avoid misapplication of exponents."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_reflect_instruction_1_2,
        "context": ["user query", thinking_1_1, answer_1_1],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1, answer_1_1], cot_reflect_instruction_1_2, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, stating resultant formula, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    critic_inst_1_2 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback_1_2, correct_1_2 = await critic_agent_1_2([taskInfo, thinking_1_2, answer_1_2], critic_inst_1_2, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_2.id}, providing feedback, thinking: {feedback_1_2.content}; answer: {correct_1_2.content}")
        if correct_1_2.content == "True":
            break
        thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2, feedback_1_2], cot_reflect_instruction_1_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, refining resultant formula statement, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    cot_reflect_instruction_1_3 = (
        "Sub-task 3: Use the factorization of x^{13} - 1 and the simplified form of f(x) to express the product \u220f_{k=0}^{12} f(omega^k) as the resultant Res(f, x^{13} - 1). "
        "Derive a closed-form expression for this resultant, carefully applying the formula from Sub-task 2 and verifying each algebraic step to avoid sign or exponentiation errors."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_reflect_instruction_1_3,
        "context": ["user query", thinking_1_2, answer_1_2, thinking_0_3, answer_0_3],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3([taskInfo, thinking_1_2, answer_1_2, thinking_0_3, answer_0_3], cot_reflect_instruction_1_3, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, deriving resultant expression, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    critic_inst_1_3 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback_1_3, correct_1_3 = await critic_agent_1_3([taskInfo, thinking_1_3, answer_1_3], critic_inst_1_3, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_3.id}, providing feedback, thinking: {feedback_1_3.content}; answer: {correct_1_3.content}")
        if correct_1_3.content == "True":
            break
        thinking_1_3, answer_1_3 = await cot_agent_1_3([taskInfo, thinking_1_2, answer_1_2, thinking_0_3, answer_0_3, thinking_1_3, answer_1_3, feedback_1_3], cot_reflect_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, refining resultant derivation, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)

    cot_reflect_instruction_1_4 = (
        "Sub-task 4: Perform algebraic validation of the resultant expression derived in Sub-task 3 by cross-checking with known polynomial identities or alternative factorization methods. "
        "Include explicit sign checks and confirm magnitude and argument of complex terms, ensuring no terms or signs are dropped."
    )
    critic_agent_1_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": cot_reflect_instruction_1_4,
        "context": ["user query", thinking_1_3, answer_1_3],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_4, answer_1_4 = await cot_agent_1_4([taskInfo, thinking_1_3, answer_1_3], cot_reflect_instruction_1_4, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_4.id}, validating resultant algebraically, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    critic_inst_1_4 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback_1_4, correct_1_4 = await critic_agent_1_4([taskInfo, thinking_1_4, answer_1_4], critic_inst_1_4, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_4.id}, providing feedback, thinking: {feedback_1_4.content}; answer: {correct_1_4.content}")
        if correct_1_4.content == "True":
            break
        thinking_1_4, answer_1_4 = await cot_agent_1_4([taskInfo, thinking_1_3, answer_1_3, thinking_1_4, answer_1_4, feedback_1_4], cot_reflect_instruction_1_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_4.id}, refining algebraic validation, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task 1.4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)

    # Stage 2: Numeric evaluation and modulo reduction
    cot_reflect_instruction_2_1 = (
        "Sub-task 1: Compute the numeric value of the resultant expression obtained in Stage 1, Sub-task 3, using binomial expansion or De Moivre's theorem as appropriate. "
        "Perform intermediate numeric sanity checks such as magnitude and argument calculations to detect sign or magnitude inconsistencies early."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", thinking_1_3, answer_1_3],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_3, answer_1_3], cot_reflect_instruction_2_1, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, computing numeric value, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    critic_inst_2_1 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback_2_1, correct_2_1 = await critic_agent_2_1([taskInfo, thinking_2_1, answer_2_1], critic_inst_2_1, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, providing feedback, thinking: {feedback_2_1.content}; answer: {correct_2_1.content}")
        if correct_2_1.content == "True":
            break
        thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_3, answer_1_3, thinking_2_1, answer_2_1, feedback_2_1], cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining numeric evaluation, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    cot_instruction_2_2 = (
        "Sub-task 2: Reduce the computed numeric value modulo 1000 to find the remainder when the product is divided by 1000. "
        "Verify the modular arithmetic steps carefully to avoid computational errors."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2([taskInfo, thinking_2_1, answer_2_1], cot_instruction_2_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_2.id}, reducing modulo 1000, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
