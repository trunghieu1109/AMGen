async def forward_92(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and organize qPCR calibration curve and Ct values

    # Sub-task 1: Analyze qPCR calibration curve parameters (efficiency=100%, R2=1, slope=-3.3)
    cot_instruction_1 = (
        "Sub-task 1: Analyze the qPCR calibration curve parameters provided (efficiency = 100%, R2 = 1, slope = -3.3) "
        "to understand the expected relationship between Ct values and target nucleic acid concentration, establishing the theoretical baseline for comparison."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing calibration curve parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Extract and organize Ct values from triplicate qPCR runs at each concentration
    cot_instruction_2 = (
        "Sub-task 2: Extract and organize the Ct values from the triplicate qPCR runs at each known concentration "
        "(100000, 10000, 1000, 100, 10 copies/µl) to prepare for quantitative comparison and evaluation of technical replicate consistency."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, extracting Ct values, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Calculate mean Ct value for each concentration from triplicates
    cot_instruction_3 = (
        "Sub-task 3: Calculate the mean Ct value for each concentration from the triplicate results "
        "to reduce variability and facilitate comparison with expected Ct values based on the calibration curve parameters."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating mean Ct values, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Evaluate Ct values and replicate consistency

    # Sub-task 4: Evaluate if observed Ct differences approximate expected 3.3 cycles per ten-fold dilution
    cot_instruction_4 = (
        "Sub-task 4: Evaluate whether the observed Ct values correspond appropriately to the known ten-fold dilutions, "
        "specifically checking if the Ct difference between each dilution step approximates the expected 3.3 cycles based on the slope and efficiency."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, evaluating Ct differences per dilution, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Assess variability within technical replicates by calculating deviation and checking if >0.3 cycles
    cot_instruction_5 = (
        "Sub-task 5: Assess the variability within technical replicates by calculating the deviation (e.g., standard deviation or range) "
        "of Ct values at each concentration and determine if deviations exceed 0.3 cycles, indicating poor replicate consistency."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking2, answer2], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, assessing replicate variability, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compare overall trend of Ct values against target nucleic acid amount to confirm inverse relationship
    cot_instruction_6 = (
        "Sub-task 6: Compare the overall trend of Ct values against the amount of target nucleic acid to verify if Ct values decrease as target concentration increases, "
        "confirming the expected inverse relationship in qPCR quantification."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking3, answer3], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, comparing Ct trend vs concentration, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Evaluate validity of statement that qPCR cannot be used for nucleic acid quantification considering all previous analyses
    cot_reflect_instruction_7 = (
        "Sub-task 7: Based on the calibration curve parameters, replicate consistency, and observed Ct trends, evaluate the validity of the statement "
        "that qPCR cannot be used for nucleic acid quantification in samples to confirm or refute this claim."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking1, answer1, thinking5, answer5, thinking6, answer6]
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, evaluating qPCR quantification validity, thinking: {thinking7.content}; answer: {answer7.content}")

    for i in range(N_max_7):
        feedback7, correct7 = await critic_agent_7([taskInfo, thinking7, answer7],
                                                  "Please review the evaluation of qPCR quantification validity and provide its limitations.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback7.content}; answer: {correct7.content}")
        if correct7.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback7])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining evaluation, thinking: {thinking7.content}; answer: {answer7.content}")

    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
